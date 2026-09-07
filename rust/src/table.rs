//! CSV ingest with pandas-equivalent dtype inference.
//!
//! `HeaderExtractor.extract_from_csv` is three lines of Python —
//! `pd.read_csv(path)` then upper-case the column names — but those three
//! lines decide the dtype of every column, and dtype is visible in the
//! acquisition CSV that AstroBin consumes: an `int64` gain renders `100`,
//! a `float64` gain renders `100.0`.
//!
//! Rules below were measured against pandas 2.2.3 with default arguments
//! (see RUST_PORT_PLAN.md hazard 14), not taken from documentation:
//!
//! | Input column                    | pandas dtype |
//! |---------------------------------|--------------|
//! | `100`, `100`                    | `int64`      |
//! | `100`, ``, `100`                | `float64`    |
//! | `100`, `None`, `100`            | `float64`    |
//! | `None`, `None`                  | `float64` (all NaN) |
//! | `abc`, `None`                   | `object` (`None` → NaN) |
//! | `100`, `1.5`                    | `float64`    |
//! | `True`, `False`                 | `bool`       |
//! | `007`, `008`                    | `int64` (→ 7, 8) |
//! | `99999999999999999999`          | `object` (i64 overflow) |
//!
//! The two that catch people: `None` is an **NA sentinel**, not the string
//! `"None"` — which matters because `[defaults]` writes a literal `None` for
//! `INSTRUME`/`TELESCOP`/`FOCNAME`/`FWHEEL`/`ROTNAME` — and a single missing
//! field anywhere demotes an integer column to float for every row.

use anyhow::{bail, Context, Result};
use std::path::Path;

/// pandas' default `na_values`, verified from `pandas._libs.parsers.STR_NA_VALUES`.
/// Matching is exact and case-sensitive (`nan` and `NaN` are listed separately;
/// `None` is listed but `NONE` is not).
const NA_VALUES: &[&str] = &[
    "", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN",
    "<NA>", "N/A", "NA", "NULL", "NaN", "None", "n/a", "nan", "null",
];

fn is_na(s: &str) -> bool {
    NA_VALUES.contains(&s)
}

/// A column's inferred type, mirroring the pandas dtype it would carry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    Int,
    Float,
    Bool,
    Str,
}

/// One cell. `Null` is pandas' NaN / NA.
#[derive(Debug, Clone, PartialEq)]
pub enum Cell {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Null,
}

impl Cell {
    pub fn is_null(&self) -> bool {
        matches!(self, Cell::Null)
    }
}

#[derive(Debug, Clone)]
pub struct Column {
    pub name: String,
    pub dtype: DType,
    pub cells: Vec<Cell>,
}

/// A parsed CSV: named columns, all of equal length.
#[derive(Debug, Clone, Default)]
pub struct Table {
    pub columns: Vec<Column>,
    pub n_rows: usize,
}

impl Table {
    pub fn column(&self, name: &str) -> Option<&Column> {
        self.columns.iter().find(|c| c.name == name)
    }

    /// Reads a CSV the way `extract_from_csv` does: infer dtypes, then
    /// upper-case every column name.
    pub fn read_csv_upper(path: &Path) -> Result<Self> {
        let mut t = Self::read_csv(path)?;
        for c in &mut t.columns {
            c.name = c.name.to_uppercase();
        }
        Ok(t)
    }

    pub fn read_csv(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading CSV {}", path.display()))?;
        Self::parse_str(&text)
    }

    pub fn parse_str(text: &str) -> Result<Self> {
        let mut records = parse_records(text);
        if records.is_empty() {
            return Ok(Table::default());
        }
        let header = records.remove(0);
        let width = header.len();

        // Transpose into raw per-column string vectors.
        let mut raw: Vec<Vec<String>> = vec![Vec::with_capacity(records.len()); width];
        for (i, rec) in records.iter().enumerate() {
            if rec.len() != width {
                bail!(
                    "row {} has {} fields, expected {width}",
                    i + 2,
                    rec.len()
                );
            }
            for (col, field) in rec.iter().enumerate() {
                raw[col].push(field.clone());
            }
        }

        let n_rows = records.len();
        let columns = header
            .into_iter()
            .zip(raw)
            .map(|(name, values)| {
                let dtype = infer_dtype(&values);
                let cells = values.iter().map(|v| coerce(v, dtype)).collect();
                Column { name, dtype, cells }
            })
            .collect();

        Ok(Table { columns, n_rows })
    }
}

/// pandas' inference order: integer (only when nothing is missing), then
/// float, then bool, else string.
fn infer_dtype(values: &[String]) -> DType {
    let has_na = values.iter().any(|v| is_na(v));
    let non_na: Vec<&String> = values.iter().filter(|v| !is_na(v)).collect();

    if non_na.is_empty() {
        // An all-NA column is float64 (all NaN), never object.
        return DType::Float;
    }

    // Integers demote to float the moment any value is missing, which is how
    // gain 100 starts rendering as 100.0.
    if !has_na && non_na.iter().all(|v| parse_int(v).is_some()) {
        return DType::Int;
    }
    // An integer literal too large for int64 does NOT become a float in
    // pandas -- the column stays object. Check before the float branch,
    // since Rust would happily parse "99999999999999999999" as 1e20.
    if non_na
        .iter()
        .any(|v| is_int_shaped(v) && parse_int(v).is_none())
    {
        return DType::Str;
    }
    if non_na.iter().all(|v| parse_float(v).is_some()) {
        return DType::Float;
    }
    if non_na.iter().all(|v| parse_bool(v).is_some()) {
        return DType::Bool;
    }
    DType::Str
}

fn coerce(value: &str, dtype: DType) -> Cell {
    if is_na(value) {
        return Cell::Null;
    }
    match dtype {
        DType::Int => parse_int(value).map(Cell::Int).unwrap_or(Cell::Null),
        DType::Float => parse_float(value).map(Cell::Float).unwrap_or(Cell::Null),
        DType::Bool => parse_bool(value).map(Cell::Bool).unwrap_or(Cell::Null),
        DType::Str => Cell::Str(value.to_string()),
    }
}

/// True when the text is an integer literal (optional sign, then digits only),
/// regardless of whether it fits in an i64.
fn is_int_shaped(s: &str) -> bool {
    let t = s.trim();
    let body = t.strip_prefix(['+', '-']).unwrap_or(t);
    !body.is_empty() && body.bytes().all(|b| b.is_ascii_digit())
}

/// Integer parse with pandas' semantics: surrounding whitespace tolerated,
/// leading zeros fine (`007` → 7), anything beyond i64 is *not* an integer
/// (pandas leaves such a column as object).
fn parse_int(s: &str) -> Option<i64> {
    let t = s.trim();
    if t.is_empty() {
        return None;
    }
    let body = t.strip_prefix(['+', '-']).unwrap_or(t);
    if body.is_empty() || !body.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    t.parse::<i64>().ok()
}

fn parse_float(s: &str) -> Option<f64> {
    let t = s.trim();
    if t.is_empty() {
        return None;
    }
    // Rust accepts "inf"/"NaN" spellings that pandas treats differently; the
    // NA sentinels are already filtered out before this is reached, and a
    // bare "inf" does parse as a float in pandas too.
    t.parse::<f64>().ok()
}

fn parse_bool(s: &str) -> Option<bool> {
    match s.trim() {
        "True" | "TRUE" | "true" => Some(true),
        "False" | "FALSE" | "false" => Some(false),
        _ => None,
    }
}

/// Splits CSV text into records, honouring RFC4180 quoting and pandas'
/// `skip_blank_lines=True` (a wholly empty line is dropped, not read as a
/// row of missing values).
fn parse_records(text: &str) -> Vec<Vec<String>> {
    let mut records = Vec::new();
    let mut field = String::new();
    let mut record: Vec<String> = Vec::new();
    let mut in_quotes = false;
    let mut chars = text.chars().peekable();

    while let Some(c) = chars.next() {
        if in_quotes {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    field.push('"');
                } else {
                    in_quotes = false;
                }
            } else {
                field.push(c);
            }
            continue;
        }
        match c {
            '"' if field.is_empty() => in_quotes = true,
            ',' => record.push(std::mem::take(&mut field)),
            '\r' => {}
            '\n' => {
                record.push(std::mem::take(&mut field));
                if !(record.len() == 1 && record[0].is_empty()) {
                    records.push(std::mem::take(&mut record));
                } else {
                    record.clear();
                }
            }
            _ => field.push(c),
        }
    }
    if !field.is_empty() || !record.is_empty() {
        record.push(field);
        if !(record.len() == 1 && record[0].is_empty()) {
            records.push(record);
        }
    }
    records
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dtype_of(csv: &str) -> DType {
        Table::parse_str(csv).unwrap().column("g").unwrap().dtype
    }

    #[test]
    fn integer_column_stays_integer() {
        assert_eq!(dtype_of("g,h\n100,1\n100,2\n"), DType::Int);
    }

    #[test]
    fn one_missing_field_demotes_integers_to_float() {
        // This is the 100 -> 100.0 failure surface.
        assert_eq!(dtype_of("g,h\n100,1\n,2\n100,3\n"), DType::Float);
        assert_eq!(dtype_of("g,h\n100,1\nNone,2\n100,3\n"), DType::Float);
        assert_eq!(dtype_of("g,h\n100,1\nNA,2\n100,3\n"), DType::Float);
    }

    #[test]
    fn none_is_a_null_sentinel_not_the_string_none() {
        // [defaults] writes a literal None for INSTRUME/TELESCOP/FOCNAME/etc.
        assert_eq!(dtype_of("g,h\nNone,1\nNone,2\n"), DType::Float);
        let t = Table::parse_str("g,h\nabc,1\nNone,2\n").unwrap();
        let g = t.column("g").unwrap();
        assert_eq!(g.dtype, DType::Str);
        assert_eq!(g.cells[0], Cell::Str("abc".into()));
        assert_eq!(g.cells[1], Cell::Null);
    }

    #[test]
    fn mixed_int_and_float_is_float() {
        assert_eq!(dtype_of("g,h\n100,1\n1.5,2\n"), DType::Float);
    }

    #[test]
    fn booleans_and_leading_zeros_and_overflow() {
        assert_eq!(dtype_of("g,h\nTrue,1\nFalse,2\n"), DType::Bool);
        let t = Table::parse_str("g,h\n007,1\n008,2\n").unwrap();
        assert_eq!(t.column("g").unwrap().dtype, DType::Int);
        assert_eq!(t.column("g").unwrap().cells[0], Cell::Int(7));
        // Beyond i64, pandas keeps the column as object.
        assert_eq!(dtype_of("g,h\n99999999999999999999,1\n2,2\n"), DType::Str);
    }

    #[test]
    fn blank_lines_are_skipped_not_read_as_missing_rows() {
        let t = Table::parse_str("g,h\n100,1\n\n100,2\n").unwrap();
        assert_eq!(t.n_rows, 2);
        assert_eq!(t.column("g").unwrap().dtype, DType::Int);
    }

    #[test]
    fn quoted_fields_may_contain_commas_and_quotes() {
        let t = Table::parse_str("a,b\n\"x,y\",\"he said \"\"hi\"\"\"\n").unwrap();
        assert_eq!(t.column("a").unwrap().cells[0], Cell::Str("x,y".into()));
        assert_eq!(
            t.column("b").unwrap().cells[0],
            Cell::Str("he said \"hi\"".into())
        );
    }

    #[test]
    fn column_names_are_upper_cased_like_extract_from_csv() {
        let mut t = Table::parse_str("imagetyp,Gain\nLIGHT,100\n").unwrap();
        for c in &mut t.columns {
            c.name = c.name.to_uppercase();
        }
        assert!(t.column("IMAGETYP").is_some());
        assert!(t.column("GAIN").is_some());
    }
}
