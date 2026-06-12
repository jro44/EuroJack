from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
from pypdf import PdfReader


MAIN_PDF_PATH = Path("./1.pdf")   # Eurojackpot 5/50
EURO_PDF_PATH = Path("./2.pdf")   # Eurojackpot 2/12

DRAW_MIN = 1
DRAW_MAX = 9999

MAIN_NUMBER_MIN = 1
MAIN_NUMBER_MAX = 50
MAIN_BALLS = 5

EURO_NUMBER_MIN = 1
EURO_NUMBER_MAX = 12
EURO_BALLS = 2

MAX_CONSECUTIVE_MAIN_CHAIN = 2
ALLOWED_MAIN_ODD_COUNTS = {2, 3}
ALLOWED_EURO_ODD_COUNTS = {0, 1, 2}


@dataclass(frozen=True)
class GameConfig:
    name: str
    number_min: int
    number_max: int
    balls_per_draw: int
    columns_prefix: str


@dataclass(frozen=True)
class ParsedDraw:
    draw_id: int
    numbers: Tuple[int, ...]


MAIN_CONFIG = GameConfig(
    name="Eurojackpot 5/50",
    number_min=MAIN_NUMBER_MIN,
    number_max=MAIN_NUMBER_MAX,
    balls_per_draw=MAIN_BALLS,
    columns_prefix="M",
)

EURO_CONFIG = GameConfig(
    name="Eurojackpot 2/12",
    number_min=EURO_NUMBER_MIN,
    number_max=EURO_NUMBER_MAX,
    balls_per_draw=EURO_BALLS,
    columns_prefix="E",
)


class MultiPaskoPdfParser:
    """
    Parser plików PDF MultiPasko.

    Format używany przez pliki:
    - na stronie najpierw występują wiersze wyników,
    - pod nimi występują numery losowań,
    - kolejność wyników odpowiada kolejności numerów losowań.

    Parser obsługuje również sklejone tokeny, np.:
    - 0102 -> 01, 02
    - 363840 -> 36, 38, 40
    """

    def __init__(self, pdf_path: Path, config: GameConfig):
        self.pdf_path = pdf_path
        self.config = config

    def exists(self) -> bool:
        return self.pdf_path.exists() and self.pdf_path.is_file()

    def split_number_token(self, token: str) -> List[int]:
        if len(token) > 2 and len(token) % 2 == 0:
            values = [int(token[i:i + 2]) for i in range(0, len(token), 2)]
            if all(self.config.number_min <= value <= self.config.number_max for value in values):
                return values
        return [int(token)]

    def parse_page(self, text: str) -> List[ParsedDraw]:
        """
        Strona MultiPasko ma dwie sekcje:
        1) wyniki,
        2) numery losowań.

        Najbezpieczniej jest więc znaleźć punkt podziału strony. To rozwiązuje
        problem dwuznacznych tokenów typu 0711 albo 1112, które w sekcji wyników
        oznaczają parę liczb, a w sekcji numerów oznaczają numer losowania.
        """
        lines = []

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if self.config.name.lower() in line.lower() or "eurojackpot" in line.lower():
                continue
            if re.findall(r"\d+", line):
                lines.append(line)

        def parse_result_line(line: str) -> Optional[Tuple[int, ...]]:
            tokens = re.findall(r"\d+", line)
            numbers: List[int] = []

            for token in tokens:
                numbers.extend(self.split_number_token(token))

            numbers = [
                number for number in numbers
                if self.config.number_min <= number <= self.config.number_max
            ]

            if (
                len(numbers) == self.config.balls_per_draw
                and len(set(numbers)) == self.config.balls_per_draw
            ):
                return tuple(sorted(numbers))

            return None

        def parse_draw_id_line(line: str) -> Optional[int]:
            tokens = re.findall(r"\d+", line)
            if len(tokens) != 1:
                return None
            token = tokens[0]
            if len(token) != 4:
                return None
            draw_id = int(token)
            if DRAW_MIN <= draw_id <= DRAW_MAX:
                return draw_id
            return None

        best_results: List[Tuple[int, ...]] = []
        best_ids: List[int] = []
        best_score = -10**9

        for split_index in range(1, len(lines)):
            result_rows = [
                parsed for parsed in (parse_result_line(line) for line in lines[:split_index])
                if parsed is not None
            ]
            draw_ids = [
                parsed for parsed in (parse_draw_id_line(line) for line in lines[split_index:])
                if parsed is not None
            ]

            if not result_rows or not draw_ids:
                continue

            pair_count = min(len(result_rows), len(draw_ids))
            score = pair_count * 100 - abs(len(result_rows) - len(draw_ids)) * 10

            # Lekka premia za naturalny układ: liczba wyników równa liczbie numerów.
            if len(result_rows) == len(draw_ids):
                score += 50

            # Lekka premia za malejące numery losowań w sekcji ID.
            if len(draw_ids) >= 2 and all(draw_ids[i] > draw_ids[i + 1] for i in range(len(draw_ids) - 1)):
                score += 20

            if score > best_score:
                best_score = score
                best_results = result_rows
                best_ids = draw_ids

        page_draws: List[ParsedDraw] = []

        for draw_id, numbers in zip(best_ids, best_results):
            page_draws.append(ParsedDraw(draw_id=draw_id, numbers=numbers))

        return page_draws

    def parse(self) -> pd.DataFrame:
        if not self.exists():
            raise FileNotFoundError(
                f'Nie znaleziono pliku "{self.pdf_path.name}". '
                "Umieść go w tym samym folderze co aplikacja."
            )

        reader = PdfReader(str(self.pdf_path))
        all_draws: Dict[int, ParsedDraw] = {}

        for page in reader.pages:
            text = page.extract_text() or ""
            for draw in self.parse_page(text):
                all_draws[draw.draw_id] = draw

        if not all_draws:
            raise ValueError(f"Nie udało się odczytać poprawnych losowań z pliku {self.pdf_path.name}.")

        rows = []

        for draw in all_draws.values():
            row: Dict[str, int] = {"Losowanie": draw.draw_id}
            for index, number in enumerate(draw.numbers, start=1):
                row[f"{self.config.columns_prefix}{index}"] = int(number)
            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.drop_duplicates(subset=["Losowanie"])
        df = df.sort_values("Losowanie", ascending=False).reset_index(drop=True)

        return df


class EurojackpotDatabaseLoader:
    """Łączy plik 5/50 i plik 2/12 po numerze losowania."""

    @staticmethod
    def load(main_pdf_path: Path, euro_pdf_path: Path) -> pd.DataFrame:
        main_df = MultiPaskoPdfParser(main_pdf_path, MAIN_CONFIG).parse()
        euro_df = MultiPaskoPdfParser(euro_pdf_path, EURO_CONFIG).parse()

        merged = pd.merge(main_df, euro_df, on="Losowanie", how="inner")
        merged = merged.drop_duplicates(subset=["Losowanie"])
        merged = merged.sort_values("Losowanie", ascending=False).reset_index(drop=True)

        if merged.empty:
            raise ValueError("Nie udało się połączyć plików 5/50 i 2/12 po numerze losowania.")

        main_columns = [f"M{i}" for i in range(1, MAIN_BALLS + 1)]
        euro_columns = [f"E{i}" for i in range(1, EURO_BALLS + 1)]

        merged["Suma_5z50"] = merged[main_columns].sum(axis=1)
        merged["Suma_2z12"] = merged[euro_columns].sum(axis=1)
        merged["Suma_całości"] = merged["Suma_5z50"] + merged["Suma_2z12"]

        return merged


class FrequencyAnalyzer:
    """Analiza częstotliwości globalnej i cyklicznej."""

    def __init__(self, df: pd.DataFrame, columns: Sequence[str], number_min: int, number_max: int):
        self.df = df
        self.columns = list(columns)
        self.number_min = number_min
        self.number_max = number_max

    def frequency_table(self, window: Optional[int] = None) -> pd.DataFrame:
        source = self.df.head(window) if window else self.df
        values = source[self.columns].to_numpy().flatten()
        counts = pd.Series(values).value_counts().reindex(
            range(self.number_min, self.number_max + 1),
            fill_value=0,
        )

        table = pd.DataFrame(
            {
                "Liczba": counts.index,
                "Wystąpienia": counts.values,
                "Procent_losowań": np.round(counts.values / max(1, len(source)) * 100, 2),
            }
        )

        hot_border = table["Wystąpienia"].quantile(0.75)
        cold_border = table["Wystąpienia"].quantile(0.25)

        table["Stan"] = "Neutralna"
        table.loc[table["Wystąpienia"] >= hot_border, "Stan"] = "Gorąca"
        table.loc[table["Wystąpienia"] <= cold_border, "Stan"] = "Zimna"

        return table.sort_values(
            ["Wystąpienia", "Liczba"],
            ascending=[False, True],
        ).reset_index(drop=True)

    def probability(self, window: Optional[int] = None) -> np.ndarray:
        table = self.frequency_table(window=window).sort_values("Liczba")
        weights = table["Wystąpienia"].to_numpy(dtype=float) + 1.0
        return weights / weights.sum()


class CoOccurrenceCouplingEngine:
    """Macierz sprzężenia par dla danego zakresu liczb."""

    def __init__(
        self,
        df: pd.DataFrame,
        columns: Sequence[str],
        number_min: int,
        number_max: int,
    ):
        self.df = df
        self.columns = list(columns)
        self.number_min = number_min
        self.number_max = number_max
        self.size = number_max - number_min + 1

    def _index(self, number: int) -> int:
        return number - self.number_min

    def pair_matrix(self, window: Optional[int] = None) -> np.ndarray:
        source = self.df.head(window) if window else self.df
        matrix = np.zeros((self.size, self.size), dtype=float)

        for row in source[self.columns].to_numpy():
            nums = sorted(map(int, row))
            for a, b in itertools.combinations(nums, 2):
                ia = self._index(a)
                ib = self._index(b)
                matrix[ia, ib] += 1.0
                matrix[ib, ia] += 1.0

        max_value = matrix.max()
        return matrix / max_value if max_value > 0 else matrix

    def coupling_factor(
        self,
        selected: Sequence[int],
        window: int = 50,
        strength: float = 1.0,
    ) -> np.ndarray:
        matrix = self.pair_matrix(window=window)
        factor = np.ones(self.size, dtype=float)

        for number in selected:
            if self.number_min <= number <= self.number_max:
                factor *= 1.0 + strength * matrix[self._index(number)]

        for number in selected:
            if self.number_min <= number <= self.number_max:
                factor[self._index(number)] = 0.0

        if factor.sum() <= 0:
            factor = np.ones(self.size, dtype=float)

        return factor / factor.sum()

    def top_pairs(self, window: int = 50, limit: int = 30) -> pd.DataFrame:
        source = self.df.head(window)
        counter: Dict[Tuple[int, int], int] = {}

        for row in source[self.columns].to_numpy():
            for pair in itertools.combinations(sorted(map(int, row)), 2):
                counter[pair] = counter.get(pair, 0) + 1

        rows = [
            {"Para": f"{pair[0]:02d}-{pair[1]:02d}", "Wystąpienia": count}
            for pair, count in counter.items()
        ]

        if not rows:
            return pd.DataFrame(columns=["Para", "Wystąpienia"])

        return pd.DataFrame(rows).sort_values(
            "Wystąpienia",
            ascending=False,
        ).head(limit).reset_index(drop=True)

    def top_triples(self, window: int = 50, limit: int = 30) -> pd.DataFrame:
        source = self.df.head(window)
        counter: Dict[Tuple[int, int, int], int] = {}

        for row in source[self.columns].to_numpy():
            if len(row) < 3:
                continue
            for triple in itertools.combinations(sorted(map(int, row)), 3):
                counter[triple] = counter.get(triple, 0) + 1

        rows = [
            {"Trójka": f"{triple[0]:02d}-{triple[1]:02d}-{triple[2]:02d}", "Wystąpienia": count}
            for triple, count in counter.items()
        ]

        if not rows:
            return pd.DataFrame(columns=["Trójka", "Wystąpienia"])

        return pd.DataFrame(rows).sort_values(
            "Wystąpienia",
            ascending=False,
        ).head(limit).reset_index(drop=True)


class KineticMemoryEngine:
    """Pamięć kinetyczna oparta na przesunięciach kolumn i następnikach Markowa."""

    def __init__(
        self,
        df: pd.DataFrame,
        columns: Sequence[str],
        number_min: int,
        number_max: int,
    ):
        self.df_newest = df.sort_values("Losowanie", ascending=False).reset_index(drop=True)
        self.columns = list(columns)
        self.number_min = number_min
        self.number_max = number_max
        self.size = number_max - number_min + 1

    def _index(self, number: int) -> int:
        return number - self.number_min

    def rolling_delta_frame(self, window: int = 30) -> pd.DataFrame:
        source = self.df_newest.head(window).sort_values("Losowanie", ascending=True).reset_index(drop=True)

        if len(source) < 2:
            return pd.DataFrame()

        previous_ids = source["Losowanie"].iloc[:-1].to_numpy()
        next_ids = source["Losowanie"].iloc[1:].to_numpy()

        deltas = source[self.columns].diff().dropna().astype(int)
        deltas.columns = [f"Delta_{column}" for column in self.columns]

        return pd.concat(
            [
                pd.DataFrame({"Z_losowania": previous_ids, "Do_losowania": next_ids}),
                deltas.reset_index(drop=True),
            ],
            axis=1,
        )

    def recency_weighted_delta_probability(self, windows: Sequence[int] = (15, 30, 50)) -> np.ndarray:
        scores = np.ones(self.size, dtype=float)
        newest_numbers = self.df_newest.loc[0, self.columns].to_numpy(dtype=int)

        for window in windows:
            source = self.df_newest.head(window).sort_values("Losowanie", ascending=True).reset_index(drop=True)

            if len(source) < 2:
                continue

            values = source[self.columns].to_numpy(dtype=int)
            deltas = np.diff(values, axis=0)

            for idx, delta_vector in enumerate(deltas):
                recency_rank = len(deltas) - idx
                weight = max(1.0, float(recency_rank))

                for position in range(len(self.columns)):
                    candidate = int(newest_numbers[position] + delta_vector[position])

                    if self.number_min <= candidate <= self.number_max:
                        scores[self._index(candidate)] += weight / max(1, window / 10)

                    for near in (candidate - 1, candidate + 1):
                        if self.number_min <= near <= self.number_max:
                            scores[self._index(near)] += 0.20 * weight / max(1, window / 10)

        return scores / scores.sum()

    def successor_probability(self, window: int = 50) -> np.ndarray:
        source = self.df_newest.head(window).sort_values("Losowanie", ascending=True).reset_index(drop=True)
        scores = np.ones(self.size, dtype=float)

        if len(source) < 2:
            return scores / scores.sum()

        latest_numbers = set(map(int, self.df_newest.loc[0, self.columns].tolist()))
        rows = source[self.columns].to_numpy(dtype=int)

        for idx in range(len(rows) - 1):
            current = set(map(int, rows[idx]))
            successor = set(map(int, rows[idx + 1]))
            recency_weight = idx + 1
            overlap_strength = len(current.intersection(latest_numbers)) + 1

            for number in successor:
                scores[self._index(number)] += recency_weight * overlap_strength * 0.15

        return scores / scores.sum()

    def final_kinetic_probability(
        self,
        recency_weight: float,
        global_probability: np.ndarray,
    ) -> np.ndarray:
        kinetic = (
            0.60 * self.recency_weighted_delta_probability()
            + 0.40 * self.successor_probability(window=50)
        )
        kinetic = kinetic / kinetic.sum()

        final = recency_weight * kinetic + (1.0 - recency_weight) * global_probability

        return final / final.sum()


class SumPredictionEngine:
    """Prognozowanie następnej sumy na podstawie historii, świeżego cyklu i trajektorii sum."""

    def __init__(self, df: pd.DataFrame):
        self.df_newest = df.sort_values("Losowanie", ascending=False).reset_index(drop=True)

    def _score_sum_series(
        self,
        sum_column: str,
        window: int = 80,
        recency_strength: float = 0.70,
    ) -> pd.DataFrame:
        source = self.df_newest.head(window).copy()
        values_newest = source[sum_column].astype(int).to_numpy()

        if len(values_newest) < 3:
            return pd.DataFrame(columns=["Suma", "Wynik_modelu", "Wystąpienia", "Odległość_od_projekcji"])

        values_oldest = values_newest[::-1]
        latest_sum = int(values_newest[0])

        deltas = np.diff(values_oldest)
        recent_delta_weights = np.linspace(1.0, 3.0, num=len(deltas))
        weighted_delta = float(np.average(deltas, weights=recent_delta_weights))
        projected_sum = latest_sum + weighted_delta

        min_sum = int(source[sum_column].min())
        max_sum = int(source[sum_column].max())

        all_sums = list(range(min_sum, max_sum + 1))
        occurrence = pd.Series(values_newest).value_counts().to_dict()

        rows = []
        for value in all_sums:
            frequency_score = occurrence.get(value, 0) + 1.0

            # Waga świeżości: im nowsze wystąpienie danej sumy, tym większe znaczenie.
            recency_score = 1.0
            for idx, historical_value in enumerate(values_newest):
                if int(historical_value) == value:
                    recency_score += (len(values_newest) - idx) / len(values_newest)

            projection_distance = abs(value - projected_sum)
            projection_score = 1.0 / (1.0 + projection_distance)

            center = float(np.mean(values_newest))
            std = float(np.std(values_newest)) or 1.0
            normality_score = np.exp(-0.5 * ((value - center) / std) ** 2)

            final_score = (
                (1.00 - recency_strength) * frequency_score
                + recency_strength * recency_score
                + 10.0 * projection_score
                + 2.0 * normality_score
            )

            rows.append(
                {
                    "Suma": int(value),
                    "Wynik_modelu": round(float(final_score), 6),
                    "Wystąpienia": int(occurrence.get(value, 0)),
                    "Odległość_od_projekcji": round(float(projection_distance), 3),
                    "Projekcja_delta": round(float(projected_sum), 3),
                    "Średnia_okna": round(float(center), 3),
                }
            )

        table = pd.DataFrame(rows)
        return table.sort_values(
            ["Wynik_modelu", "Wystąpienia"],
            ascending=[False, False],
        ).reset_index(drop=True)

    def predict(self, mode: str = "Suma_całości", window: int = 80) -> Tuple[int, pd.DataFrame]:
        table = self._score_sum_series(mode, window=window)
        if table.empty:
            return 0, table
        return int(table.iloc[0]["Suma"]), table

    def all_predictions(self, window: int = 80) -> pd.DataFrame:
        labels = {
            "Suma_5z50": "Suma 5/50",
            "Suma_2z12": "Suma 2/12",
            "Suma_całości": "Suma całości",
        }

        rows = []

        for column, label in labels.items():
            predicted, table = self.predict(column, window=window)

            if table.empty:
                continue

            top_range = table.head(5)["Suma"].astype(int).tolist()

            rows.append(
                {
                    "Typ": label,
                    "Najmocniejsza_suma": predicted,
                    "TOP_5_sąsiednich_możliwości": ", ".join(str(x) for x in top_range),
                    "Projekcja_delta": float(table.iloc[0]["Projekcja_delta"]),
                    "Średnia_okna": float(table.iloc[0]["Średnia_okna"]),
                }
            )

        return pd.DataFrame(rows)


class EurojackpotRealityFilter:
    """Filtry rzeczywistości dla kuponu Eurojackpot 5/50 + 2/12."""

    @staticmethod
    def valid_odd_even(numbers: Tuple[int, ...], allowed_odd_counts: set[int]) -> bool:
        return sum(1 for number in numbers if number % 2 == 1) in allowed_odd_counts

    @staticmethod
    def valid_consecutive_blocks(numbers: Tuple[int, ...], max_chain: int) -> bool:
        sorted_numbers = sorted(numbers)
        longest = 1
        current = 1

        for index in range(1, len(sorted_numbers)):
            if sorted_numbers[index] == sorted_numbers[index - 1] + 1:
                current += 1
                longest = max(longest, current)
            else:
                current = 1

        return longest <= max_chain

    @classmethod
    def accept_main(cls, numbers: Tuple[int, ...]) -> bool:
        return (
            cls.valid_odd_even(numbers, ALLOWED_MAIN_ODD_COUNTS)
            and cls.valid_consecutive_blocks(numbers, MAX_CONSECUTIVE_MAIN_CHAIN)
        )

    @classmethod
    def accept_euro(cls, numbers: Tuple[int, ...]) -> bool:
        return cls.valid_odd_even(numbers, ALLOWED_EURO_ODD_COUNTS)


class SinglePoolPredictionEngine:
    """Silnik predykcyjny dla jednej puli liczb."""

    def __init__(
        self,
        df: pd.DataFrame,
        columns: Sequence[str],
        number_min: int,
        number_max: int,
    ):
        self.df = df
        self.columns = list(columns)
        self.number_min = number_min
        self.number_max = number_max

        self.frequency = FrequencyAnalyzer(df, columns, number_min, number_max)
        self.kinetic = KineticMemoryEngine(df, columns, number_min, number_max)
        self.cooccurrence = CoOccurrenceCouplingEngine(df, columns, number_min, number_max)

    def model_probability_table(self, rolling_window: int, recency_weight: float) -> pd.DataFrame:
        global_probability = self.frequency.probability(window=None)
        rolling_probability = self.frequency.probability(window=rolling_window)

        mixed_global = 0.55 * global_probability + 0.45 * rolling_probability
        mixed_global = mixed_global / mixed_global.sum()

        final = self.kinetic.final_kinetic_probability(
            recency_weight=recency_weight,
            global_probability=mixed_global,
        )

        return pd.DataFrame(
            {
                "Liczba": range(self.number_min, self.number_max + 1),
                "Waga_globalna_i_cykl": np.round(mixed_global, 6),
                "Waga_kinetyczna": np.round(final, 6),
                "Waga_końcowa": np.round(final, 6),
            }
        ).sort_values("Waga_końcowa", ascending=False).reset_index(drop=True)

    @staticmethod
    def _weighted_pick(available: Sequence[int], weights_by_number: Dict[int, float]) -> int:
        available = sorted(set(int(number) for number in available))
        local_weights = np.array(
            [max(0.000001, weights_by_number[number]) for number in available],
            dtype=float,
        )
        local_weights = local_weights / local_weights.sum()
        return int(np.random.choice(available, p=local_weights))

    def sample_one(
        self,
        balls: int,
        rolling_window: int,
        recency_weight: float,
        coupling_strength: float,
    ) -> Tuple[int, ...]:
        table = self.model_probability_table(rolling_window, recency_weight)
        weight_map = {
            int(row["Liczba"]): float(row["Waga_końcowa"])
            for _, row in table.iterrows()
        }

        selected: List[int] = []
        available = list(range(self.number_min, self.number_max + 1))

        while len(selected) < balls:
            coupling = self.cooccurrence.coupling_factor(
                selected,
                window=rolling_window,
                strength=coupling_strength,
            )

            dynamic_weight_map: Dict[int, float] = {}

            for number in available:
                idx = number - self.number_min
                base = weight_map[number]
                dynamic_weight_map[number] = base * (1.0 + coupling_strength * coupling[idx] * len(coupling))

            chosen = self._weighted_pick(available, dynamic_weight_map)
            selected.append(chosen)
            available.remove(chosen)

        return tuple(sorted(selected))


class EurojackpotPredictionEngine:
    """Główny silnik Eurojackpot: 5/50 + 2/12 + suma zadana + prognoza sumy."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.main_columns = [f"M{i}" for i in range(1, MAIN_BALLS + 1)]
        self.euro_columns = [f"E{i}" for i in range(1, EURO_BALLS + 1)]

        self.main_engine = SinglePoolPredictionEngine(
            df,
            self.main_columns,
            MAIN_NUMBER_MIN,
            MAIN_NUMBER_MAX,
        )

        self.euro_engine = SinglePoolPredictionEngine(
            df,
            self.euro_columns,
            EURO_NUMBER_MIN,
            EURO_NUMBER_MAX,
        )

        self.sum_engine = SumPredictionEngine(df)

    def _target_sum_ok(
        self,
        main_numbers: Tuple[int, ...],
        euro_numbers: Tuple[int, ...],
        target_sum: Optional[int],
        target_mode: str,
    ) -> bool:
        if target_sum is None:
            return True

        main_sum = sum(main_numbers)
        euro_sum = sum(euro_numbers)
        total_sum = main_sum + euro_sum

        if target_mode == "Suma całości":
            return total_sum == target_sum

        if target_mode == "Suma 5/50":
            return main_sum == target_sum

        if target_mode == "Suma 2/12":
            return euro_sum == target_sum

        return True

    def generate(
        self,
        count: int,
        rolling_window: int,
        recency_weight: float,
        coupling_strength: float,
        target_sum: Optional[int] = None,
        target_mode: str = "Suma całości",
    ) -> pd.DataFrame:
        rows: List[Dict[str, int | str]] = []
        used: set[Tuple[int, ...]] = set()

        attempts = 0
        max_attempts = max(100000, count * 20000)

        while len(rows) < count and attempts < max_attempts:
            attempts += 1

            main_numbers = self.main_engine.sample_one(
                balls=MAIN_BALLS,
                rolling_window=rolling_window,
                recency_weight=recency_weight,
                coupling_strength=coupling_strength,
            )

            euro_numbers = self.euro_engine.sample_one(
                balls=EURO_BALLS,
                rolling_window=rolling_window,
                recency_weight=recency_weight,
                coupling_strength=coupling_strength,
            )

            if not EurojackpotRealityFilter.accept_main(main_numbers):
                continue

            if not EurojackpotRealityFilter.accept_euro(euro_numbers):
                continue

            if not self._target_sum_ok(main_numbers, euro_numbers, target_sum, target_mode):
                continue

            unique_key = main_numbers + euro_numbers

            if unique_key in used:
                continue

            used.add(unique_key)

            main_odd = sum(1 for number in main_numbers if number % 2 == 1)
            euro_odd = sum(1 for number in euro_numbers if number % 2 == 1)

            row: Dict[str, int | str] = {
                "Kupon": len(rows) + 1,
                "M1": main_numbers[0],
                "M2": main_numbers[1],
                "M3": main_numbers[2],
                "M4": main_numbers[3],
                "M5": main_numbers[4],
                "E1": euro_numbers[0],
                "E2": euro_numbers[1],
                "Suma_5z50": sum(main_numbers),
                "Suma_2z12": sum(euro_numbers),
                "Suma_całości": sum(main_numbers) + sum(euro_numbers),
                "Balans_5z50": f"{main_odd}:{MAIN_BALLS - main_odd}",
                "Balans_2z12": f"{euro_odd}:{EURO_BALLS - euro_odd}",
            }

            rows.append(row)

        return pd.DataFrame(rows)


class EurojackpotBlanketRenderer:
    """Renderer blankietu 5/50 i 2/12."""

    @staticmethod
    def render_pool(
        selected: Sequence[int],
        title: str,
        number_min: int,
        number_max: int,
        columns: int,
        cell_size: int = 42,
    ) -> str:
        selected_set = set(int(number) for number in selected)
        cells = []

        for number in range(number_min, number_max + 1):
            css = "selected" if number in selected_set else ""
            cells.append(f'<div class="ej-cell {css}">{number:02d}</div>')

        return f"""
        <div class="ej-pool">
            <div class="ej-title">{title}</div>
            <div class="ej-grid" style="grid-template-columns: repeat({columns}, {cell_size}px);">
                {''.join(cells)}
            </div>
        </div>
        """

    @staticmethod
    def render(main_numbers: Sequence[int], euro_numbers: Sequence[int], title: str) -> str:
        main_html = EurojackpotBlanketRenderer.render_pool(
            main_numbers,
            "Główne liczby 5/50",
            MAIN_NUMBER_MIN,
            MAIN_NUMBER_MAX,
            columns=10,
            cell_size=40,
        )

        euro_html = EurojackpotBlanketRenderer.render_pool(
            euro_numbers,
            "Euronumery 2/12",
            EURO_NUMBER_MIN,
            EURO_NUMBER_MAX,
            columns=6,
            cell_size=40,
        )

        return f"""
        <div class="ej-wrapper">
            <div class="ej-main-title">{title}</div>
            {main_html}
            {euro_html}
        </div>

        <style>
        .ej-wrapper {{
            background: linear-gradient(145deg, #08111f, #111827);
            border: 1px solid #334155;
            border-radius: 18px;
            padding: 18px;
            margin: 16px 0 28px 0;
            max-width: 560px;
            box-shadow: 0 18px 42px rgba(0,0,0,0.35);
        }}
        .ej-main-title {{
            color: #f9fafb;
            font-weight: 900;
            font-size: 18px;
            margin-bottom: 16px;
            letter-spacing: 0.2px;
        }}
        .ej-title {{
            color: #cbd5e1;
            font-weight: 800;
            font-size: 14px;
            margin: 12px 0 10px 0;
        }}
        .ej-grid {{
            display: grid;
            gap: 7px;
        }}
        .ej-cell {{
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background: #1f2937;
            border: 1px solid #4b5563;
            color: #d1d5db;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 900;
            font-size: 13px;
        }}
        .ej-cell.selected {{
            background: radial-gradient(circle at 30% 30%, #fff7ed, #fbbf24 38%, #dc2626 100%);
            color: #111827;
            border: 2px solid #fde68a;
            box-shadow: 0 0 18px rgba(251, 191, 36, 0.95), 0 0 34px rgba(220, 38, 38, 0.55);
            transform: scale(1.08);
        }}
        </style>
        """


class EurojackpotStreamlitApp:
    """Profesjonalna aplikacja Streamlit dla Eurojackpot 5/50 + 2/12."""

    def __init__(self):
        st.set_page_config(
            page_title="Cyber-Maszyna Eurojackpot 5/50 + 2/12",
            page_icon="🎯",
            layout="wide",
        )
        self.df = self.load_database()
        self.engine = EurojackpotPredictionEngine(self.df)

    @staticmethod
    @st.cache_data(show_spinner="Wczytywanie i parsowanie plików PDF 1.pdf oraz 2.pdf...")
    def cached_load(main_path: str, euro_path: str) -> pd.DataFrame:
        return EurojackpotDatabaseLoader.load(Path(main_path), Path(euro_path))

    def load_database(self) -> pd.DataFrame:
        missing_files = []

        if not MAIN_PDF_PATH.exists():
            missing_files.append(MAIN_PDF_PATH.name)

        if not EURO_PDF_PATH.exists():
            missing_files.append(EURO_PDF_PATH.name)

        if missing_files:
            st.error(
                "Brakuje plików PDF: "
                + ", ".join(missing_files)
                + ". Umieść pliki w tym samym folderze co aplikacja."
            )
            st.stop()

        try:
            return self.cached_load(str(MAIN_PDF_PATH), str(EURO_PDF_PATH))
        except Exception as error:
            st.error(f"Nie udało się odczytać i połączyć plików PDF: {error}")
            st.stop()

    def render_header(self) -> None:
        st.title("🎯 Cyber-Maszyna Eurojackpot 5/50 + 2/12")
        st.info(
            "Aplikacja czyta dwa pliki PDF: 1.pdf jako Eurojackpot 5/50 oraz 2.pdf jako Eurojackpot 2/12. "
            "Silnik łączy losowania po numerze, analizuje częstotliwości, świeże cykle, wektory przesunięć, "
            "sprzężenia par oraz przewiduje najbardziej logiczny kierunek następnej sumy. "
            "To narzędzie statystyczne i symulacyjne — nie gwarantuje wyniku losowania."
        )

    def render_generator_tab(self) -> None:
        st.header("🚀 Generator Cyber-Maszyny")
        st.info(
            "Generator tworzy pełny kupon Eurojackpot: 5 liczb z puli 1–50 oraz 2 euronumery z puli 1–12. "
            "Możesz generować normalnie albo wymusić konkretną sumę: sumę 5/50, sumę 2/12 albo sumę całości."
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            count = st.slider("Liczba kuponów", 1, 10, 5)

        with c2:
            rolling_window = st.select_slider("Okno cyklu", options=[15, 30, 50, 80, 120], value=50)

        with c3:
            recency_weight = st.slider("Nacisk na świeży cykl", 0.10, 0.95, 0.75, 0.05)

        with c4:
            coupling_strength = st.slider("Sprzężenie par", 0.10, 2.00, 1.00, 0.10)

        st.divider()

        sum_col_1, sum_col_2, sum_col_3 = st.columns(3)

        with sum_col_1:
            use_target_sum = st.checkbox("🎯 Generuj pod konkretną sumę", value=False)

        with sum_col_2:
            target_mode = st.selectbox(
                "Którą sumę wymusić?",
                ["Suma całości", "Suma 5/50", "Suma 2/12"],
                index=0,
            )

        with sum_col_3:
            predicted_total, _ = self.engine.sum_engine.predict("Suma_całości", window=80)
            default_sum = int(predicted_total) if predicted_total else 140
            target_sum_input = st.number_input(
                "Podaj sumę",
                min_value=3,
                max_value=263,
                value=default_sum,
                step=1,
            )

        target_sum = int(target_sum_input) if use_target_sum else None

        if st.button("⚙️ URUCHOM SYMULACJĘ EUROJACKPOT", use_container_width=True, type="primary"):
            result = self.engine.generate(
                count=count,
                rolling_window=rolling_window,
                recency_weight=recency_weight,
                coupling_strength=coupling_strength,
                target_sum=target_sum,
                target_mode=target_mode,
            )

            if result.empty:
                st.error(
                    "Symulacja nie znalazła kuponów spełniających wszystkie filtry. "
                    "Spróbuj zmienić sumę, tryb sumy, okno cyklu albo sprzężenie par."
                )
                return

            st.success("Symulacja zakończona.")
            st.subheader("✅ Wynik symulacji")
            st.dataframe(result, use_container_width=True, hide_index=True)

            lines = []

            for _, row in result.iterrows():
                main_numbers = [int(row[f"M{i}"]) for i in range(1, MAIN_BALLS + 1)]
                euro_numbers = [int(row[f"E{i}"]) for i in range(1, EURO_BALLS + 1)]
                line = (
                    f"Kupon {int(row['Kupon'])}: "
                    + " ".join(f"{number:02d}" for number in main_numbers)
                    + " + "
                    + " ".join(f"{number:02d}" for number in euro_numbers)
                    + f" | suma 5/50={int(row['Suma_5z50'])}"
                    + f" | suma 2/12={int(row['Suma_2z12'])}"
                    + f" | suma całości={int(row['Suma_całości'])}"
                    + f" | balans 5/50={row['Balans_5z50']}"
                    + f" | balans 2/12={row['Balans_2z12']}"
                )
                lines.append(line)

            text = "\n".join(lines)

            st.text_area("Kopiuj kupony do schowka", value=text, height=170)

            st.download_button(
                "⬇️ Pobierz kupony TXT",
                data=text.encode("utf-8"),
                file_name="cyber_kupony_eurojackpot.txt",
                mime="text/plain",
                use_container_width=True,
            )

            st.download_button(
                "⬇️ Pobierz kupony CSV",
                data=result.to_csv(index=False).encode("utf-8"),
                file_name="cyber_kupony_eurojackpot.csv",
                mime="text/csv",
                use_container_width=True,
            )

            st.subheader("🎫 Interaktywne blankiety")
            for _, row in result.iterrows():
                main_numbers = [int(row[f"M{i}"]) for i in range(1, MAIN_BALLS + 1)]
                euro_numbers = [int(row[f"E{i}"]) for i in range(1, EURO_BALLS + 1)]
                title = (
                    f"Kupon {int(row['Kupon'])}: "
                    + " ".join(f"{number:02d}" for number in main_numbers)
                    + " + "
                    + " ".join(f"{number:02d}" for number in euro_numbers)
                )
                st.markdown(
                    EurojackpotBlanketRenderer.render(main_numbers, euro_numbers, title),
                    unsafe_allow_html=True,
                )

        st.subheader("Wagi modelu — 5/50")
        st.dataframe(
            self.engine.main_engine.model_probability_table(rolling_window, recency_weight),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Wagi modelu — 2/12")
        st.dataframe(
            self.engine.euro_engine.model_probability_table(rolling_window, recency_weight),
            use_container_width=True,
            hide_index=True,
        )

    def render_sum_prediction_tab(self) -> None:
        st.header("🔮 Predykcja następnej sumy")
        st.info(
            "Ten moduł nie losuje kuponu. On analizuje historię sum, świeże przesunięcia i powtarzalność zakresów. "
            "Wyniki możesz potem użyć w generatorze jako sumę zadaną."
        )

        c1, c2 = st.columns(2)

        with c1:
            window = st.select_slider("Okno predykcji sumy", options=[30, 50, 80, 120, 200], value=80)

        with c2:
            sum_mode_label = st.selectbox(
                "Co przewidywać?",
                ["Suma całości", "Suma 5/50", "Suma 2/12"],
                index=0,
            )

        column_map = {
            "Suma całości": "Suma_całości",
            "Suma 5/50": "Suma_5z50",
            "Suma 2/12": "Suma_2z12",
        }

        selected_column = column_map[sum_mode_label]
        predicted_sum, prediction_table = self.engine.sum_engine.predict(selected_column, window=window)

        overview = self.engine.sum_engine.all_predictions(window=window)

        st.subheader("📌 Najważniejsze wskazania")
        st.dataframe(overview, use_container_width=True, hide_index=True)

        if predicted_sum:
            st.metric(f"Najmocniejsza prognozowana suma — {sum_mode_label}", predicted_sum)

        st.subheader(f"TOP 30 sum według modelu — {sum_mode_label}")
        st.dataframe(prediction_table.head(30), use_container_width=True, hide_index=True)

        sums_history = self.df[["Losowanie", "Suma_5z50", "Suma_2z12", "Suma_całości"]].head(window)
        st.subheader("Historia sum w wybranym oknie")
        st.line_chart(sums_history.sort_values("Losowanie").set_index("Losowanie"))

    def render_archive_tab(self) -> None:
        st.header("📄 Archiwum połączonych losowań")
        st.info("Najnowsze losowanie jest na górze. Dane pochodzą z połączenia 1.pdf i 2.pdf po numerze losowania.")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Liczba połączonych losowań", len(self.df))
        c2.metric("Najnowsze", int(self.df["Losowanie"].max()))
        c3.metric("Najstarsze", int(self.df["Losowanie"].min()))
        c4.metric("Średnia suma całości", round(float(self.df["Suma_całości"].mean()), 2))

        st.dataframe(self.df, use_container_width=True, hide_index=True)

        st.download_button(
            "⬇️ Pobierz połączone archiwum CSV",
            data=self.df.to_csv(index=False).encode("utf-8"),
            file_name="archiwum_eurojackpot_polaczone.csv",
            mime="text/csv",
            use_container_width=True,
        )

    def render_cycles_tab(self) -> None:
        st.header("🔥❄️ Cykle, gorące liczby i sprzężenia")
        window = st.select_slider("Okno analizy cyklu", options=[15, 30, 50, 80, 120], value=50)

        main_frequency = FrequencyAnalyzer(
            self.df,
            [f"M{i}" for i in range(1, MAIN_BALLS + 1)],
            MAIN_NUMBER_MIN,
            MAIN_NUMBER_MAX,
        )

        euro_frequency = FrequencyAnalyzer(
            self.df,
            [f"E{i}" for i in range(1, EURO_BALLS + 1)],
            EURO_NUMBER_MIN,
            EURO_NUMBER_MAX,
        )

        main_table = main_frequency.frequency_table(window=window)
        euro_table = euro_frequency.frequency_table(window=window)

        c1, c2 = st.columns(2)

        with c1:
            st.subheader(f"Stan 5/50 w ostatnich {window} losowaniach")
            st.dataframe(main_table, use_container_width=True, hide_index=True)
            st.bar_chart(main_table.sort_values("Liczba").set_index("Liczba")["Wystąpienia"])

        with c2:
            st.subheader(f"Stan 2/12 w ostatnich {window} losowaniach")
            st.dataframe(euro_table, use_container_width=True, hide_index=True)
            st.bar_chart(euro_table.sort_values("Liczba").set_index("Liczba")["Wystąpienia"])

        main_co = CoOccurrenceCouplingEngine(
            self.df,
            [f"M{i}" for i in range(1, MAIN_BALLS + 1)],
            MAIN_NUMBER_MIN,
            MAIN_NUMBER_MAX,
        )

        euro_co = CoOccurrenceCouplingEngine(
            self.df,
            [f"E{i}" for i in range(1, EURO_BALLS + 1)],
            EURO_NUMBER_MIN,
            EURO_NUMBER_MAX,
        )

        p1, p2, p3 = st.columns(3)

        with p1:
            st.subheader("Najczęstsze pary 5/50")
            st.dataframe(main_co.top_pairs(window=window, limit=30), use_container_width=True, hide_index=True)

        with p2:
            st.subheader("Najczęstsze trójki 5/50")
            st.dataframe(main_co.top_triples(window=window, limit=30), use_container_width=True, hide_index=True)

        with p3:
            st.subheader("Najczęstsze pary 2/12")
            st.dataframe(euro_co.top_pairs(window=window, limit=30), use_container_width=True, hide_index=True)

    def render_vectors_tab(self) -> None:
        st.header("📈 Wektory przesunięć 5/50 i 2/12")
        window = st.select_slider("Okno wektorów", options=[15, 30, 50, 80], value=30)

        main_kinetic = KineticMemoryEngine(
            self.df,
            [f"M{i}" for i in range(1, MAIN_BALLS + 1)],
            MAIN_NUMBER_MIN,
            MAIN_NUMBER_MAX,
        )

        euro_kinetic = KineticMemoryEngine(
            self.df,
            [f"E{i}" for i in range(1, EURO_BALLS + 1)],
            EURO_NUMBER_MIN,
            EURO_NUMBER_MAX,
        )

        c1, c2 = st.columns(2)

        with c1:
            st.subheader(f"Macierz przesunięć 5/50 — ostatnie {window}")
            main_deltas = main_kinetic.rolling_delta_frame(window=window)
            st.dataframe(main_deltas, use_container_width=True, hide_index=True)

            if not main_deltas.empty:
                delta_columns = [f"Delta_M{i}" for i in range(1, MAIN_BALLS + 1)]
                st.line_chart(main_deltas.set_index("Do_losowania")[delta_columns])

        with c2:
            st.subheader(f"Macierz przesunięć 2/12 — ostatnie {window}")
            euro_deltas = euro_kinetic.rolling_delta_frame(window=window)
            st.dataframe(euro_deltas, use_container_width=True, hide_index=True)

            if not euro_deltas.empty:
                delta_columns = [f"Delta_E{i}" for i in range(1, EURO_BALLS + 1)]
                st.line_chart(euro_deltas.set_index("Do_losowania")[delta_columns])

    def run(self) -> None:
        self.render_header()

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                "🚀 Generator",
                "🔮 Predykcja sumy",
                "📄 Archiwum",
                "🔥❄️ Cykle",
                "📈 Wektory",
            ]
        )

        with tab1:
            self.render_generator_tab()

        with tab2:
            self.render_sum_prediction_tab()

        with tab3:
            self.render_archive_tab()

        with tab4:
            self.render_cycles_tab()

        with tab5:
            self.render_vectors_tab()


if __name__ == "__main__":
    EurojackpotStreamlitApp().run()
