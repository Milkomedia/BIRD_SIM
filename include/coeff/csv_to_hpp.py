from pathlib import Path
import csv

CSV_FILES = (
    "naca_CD.csv",
    "naca_CL.csv",
    "naca_CM.csv",
    "s20_CD.csv",
    "s20_CL.csv",
    "s20_CM.csv",
    "s40_CD.csv",
    "s40_CL.csv",
    "s40_CM.csv",
)


def cpp_number(text: str) -> str:
    text = text.strip()
    if "." not in text and "e" not in text.lower(): text += ".0"
    return text


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.reader(file))

    re_values = [
        cpp_number(value)
        for value in rows[0][1:]
    ]

    alpha_values = [
        cpp_number(row[0])
        for row in rows[1:]
    ]

    table = [
        [cpp_number(value) for value in row[1:]]
        for row in rows[1:]
    ]

    return alpha_values, re_values, table


def write_array(file, name: str, values):
    file.write(
        f"inline constexpr double {name}[{len(values)}] = {{\n"
        f"    {', '.join(values)}\n"
        "};\n\n"
    )


def write_table(file, name: str, table):
    file.write(
        f"inline constexpr double {name}"
        f"[ALPHA_COUNT][RE_COUNT] = {{\n"
    )

    for row in table:
        file.write(f"    {{{', '.join(row)}}},\n")

    file.write("};\n\n")


def main():
    coeff_dir = Path(__file__).resolve().parent
    output_path = coeff_dir / "coeff.hpp"

    tables = {}
    alpha_values = []
    re_values = []

    for index, csv_name in enumerate(CSV_FILES):
        alpha, reynolds, table = read_csv(
            coeff_dir / csv_name
        )

        if index == 0:
            alpha_values = alpha
            re_values = reynolds

        table_name = Path(csv_name).stem.upper()
        tables[table_name] = table

    with output_path.open("w", encoding="utf-8") as file:
        file.write(
            "#pragma once\n\n"
            "#include <cstddef>\n\n"
            "namespace param::coeff {\n\n"
            f"inline constexpr std::size_t ALPHA_COUNT = "
            f"{len(alpha_values)};\n"
            f"inline constexpr std::size_t RE_COUNT = "
            f"{len(re_values)};\n\n"
        )

        write_array(file, "ALPHA_VALUES", alpha_values)
        write_array(file, "RE_VALUES", re_values)

        for name, table in tables.items():
            write_table(file, name, table)

        file.write("}  // namespace param::coeff\n")

    print(f"Generated: {output_path}")


if __name__ == "__main__":
    main()