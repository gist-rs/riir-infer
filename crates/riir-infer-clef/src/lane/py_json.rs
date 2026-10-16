//! Python-style JSON rendering — the `ClefJSON` port (plan 619 T1.4).
//!
//! The clef record encoder renders the `state` and the option semantics
//! with Python's `json.dumps(value, ensure_ascii=False,
//! separators=(",", ":"), sort_keys=True)` — and floats through Python's
//! `repr(float)`. That rendering is part of the TRAINING-time token
//! stream: the head averages over instruction/option token spans, so a
//! byte of drift here moves the spans and desyncs the decision. The
//! pinned semantics (transcribed from FluidUse `ClefJSON` @ `0230049c`,
//! itself a port of Cloudflare's `joint_schema_model.py`):
//!
//! - top-level strings render AS THEMSELVES (`render`), nested strings
//!   are quoted;
//! - object keys sort (code-point order — Python's `sort_keys`);
//! - `,`/`:` separators, no spaces;
//! - `ensure_ascii=False`: non-ASCII stays raw; only `"`, `\`, and the
//!   control characters (`\b` `\f` `\n` `\r` `\t`, other `< 0x20` as
//!   `\u00xx` lowercase) escape;
//! - integers stay integers (`3`), floats carry Python `repr` form
//!   (`3.0`, `1e+16`, `1e-05`) — [`python_float`];
//! - `NaN` / `Infinity` / `-Infinity` spellings (not reachable through
//!   [`serde_json::Value`], which refuses non-finite floats — the
//!   function is public because the head-side math may render one day
//!   and the spelling is part of the contract).
//!
//! The expected strings in the tests were generated with real CPython
//! (`json.dumps` / `repr`) — an independent oracle, not this port.

use std::collections::BTreeMap;

use serde_json::Value;

/// Render `value` the way the record encoder renders a field: a
/// top-level string is inserted AS ITSELF (unquoted) — every other
/// value goes through the JSON dumper.
pub fn render(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => dump(other),
    }
}

/// `json.dumps(value, ensure_ascii=False, separators=(",", ":"),
/// sort_keys=True)` — strings quoted, keys sorted, compact separators.
pub fn dump(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(true) => "true".to_string(),
        Value::Bool(false) => "false".to_string(),
        Value::Number(n) => {
            if n.is_f64() {
                python_float(n.as_f64().expect("is_f64 guarantees as_f64"))
            } else if n.is_u64() {
                n.as_u64().expect("is_u64 guarantees as_u64").to_string()
            } else {
                n.as_i64().expect("i64 is the only remaining store").to_string()
            }
        }
        Value::String(s) => quote(s),
        Value::Array(items) => {
            let mut out = String::from("[");
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&dump(item));
            }
            out.push(']');
            out
        }
        Value::Object(map) => {
            // BTreeMap: code-point-sorted keys regardless of the
            // serde_json Map's own ordering features (Python sort_keys).
            let sorted: BTreeMap<&str, &Value> =
                map.iter().map(|(k, v)| (k.as_str(), v)).collect();
            let mut out = String::from("{");
            for (i, (k, v)) in sorted.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&quote(k));
                out.push(':');
                out.push_str(&dump(v));
            }
            out.push('}');
            out
        }
    }
}

/// Python's `repr(float)`: shortest round-trip digits, always a decimal
/// point or an exponent — fixed notation unless the decimal exponent
/// passes the ±16/10⁻⁴ window (`1e+16`, `1e-05`), exponent spelled
/// `e±NN` (at least two digits), `NaN` / `Infinity` / `-Infinity`.
pub fn python_float(value: f64) -> String {
    if value.is_nan() {
        return "NaN".to_string();
    }
    if value.is_infinite() {
        return if value > 0.0 {
            "Infinity".to_string()
        } else {
            "-Infinity".to_string()
        };
    }
    // Rust's LowerExp prints the same shortest round-trip digits as
    // CPython's dtoa; only the presentation differs (Rust never uses
    // fixed/exponent the way Python draws the line), so take the digits
    // + exponent from `{:e}` and re-present.
    let sci = format!("{value:e}");
    let (mantissa, exponent) = sci.split_once('e').expect("LowerExp always emits 'e'");
    let (negative, mantissa) = match mantissa.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, mantissa),
    };
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let exponent: i32 = exponent.parse().expect("LowerExp exponent is decimal");
    let decpt = exponent + 1; // digits before the decimal point (fixed form)

    let body = if !(-3..=16).contains(&decpt) {
        // scientific: d[.ddd]e±NN
        let mut out = String::with_capacity(digits.len() + 6);
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push('e');
        out.push(if exponent < 0 { '-' } else { '+' });
        let magnitude = exponent.unsigned_abs();
        if magnitude < 10 {
            out.push('0');
        }
        out.push_str(&magnitude.to_string());
        out
    } else if decpt <= 0 {
        format!("0.{}{digits}", "0".repeat((-decpt) as usize))
    } else if decpt as usize >= digits.len() {
        let zeros = "0".repeat(decpt as usize - digits.len());
        format!("{digits}{zeros}.0")
    } else {
        let point = decpt as usize;
        format!("{}.{}", &digits[..point], &digits[point..])
    };
    if negative {
        format!("-{body}")
    } else {
        body
    }
}

/// JSON string quoting with `ensure_ascii=False`: the six short escapes,
/// other control characters as `\u00xx` (lowercase hex), everything
/// else — including all non-ASCII — raw.
fn quote(string: &str) -> String {
    let mut out = String::with_capacity(string.len() + 2);
    out.push('"');
    for c in string.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0C}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The float battery — every expected string generated with real
    /// CPython `repr` (2026-10-17, python3 on this box). Covers the
    /// fixed/scientific boundary both sides (1e15 fixed vs 1e16 sci,
    /// 1e-4 fixed vs 1e-5 sci), the `.0` suffix, the `e±NN` padding,
    /// subnormals, and f64 extremes.
    #[test]
    fn python_float_matches_cpython_repr() {
        let cases: &[(f64, &str)] = &[
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1.0, "1.0"),
            (-1.0, "-1.0"),
            (0.5, "0.5"),
            (12.5, "12.5"),
            (1250.0, "1250.0"),
            (100.0, "100.0"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (1e-4, "0.0001"),
            (1e-5, "1e-05"),
            (0.1, "0.1"),
            (0.98, "0.98"),
            (1_234_567_890_123_456.0, "1234567890123456.0"),
            (12_345_678_901_234_568.0, "1.2345678901234568e+16"),
            (1.5e300, "1.5e+300"),
            (5e-324, "5e-324"),
            (1.797_693_134_862_315_7e308, "1.7976931348623157e+308"),
            (2.225_073_858_507_201_4e-308, "2.2250738585072014e-308"),
            (1.2345678, "1.2345678"),
            (1234.5678, "1234.5678"),
            (1e100, "1e+100"),
            (2.5e-10, "2.5e-10"),
        ];
        for (value, expected) in cases {
            assert_eq!(&python_float(*value), expected, "value {value:e}");
        }
        assert_eq!(python_float(f64::NAN), "NaN");
        assert_eq!(python_float(f64::INFINITY), "Infinity");
        assert_eq!(python_float(f64::NEG_INFINITY), "-Infinity");
    }

    /// Full dumps vs real CPython `json.dumps(..., ensure_ascii=False,
    /// separators=(",", ":"), sort_keys=True)` output — nested objects,
    /// arrays, bools, the int/float distinction, escapes, non-ASCII.
    #[test]
    fn dump_matches_cpython_json_dumps() {
        // {"balance":1250.0,"big":1000000000000000.0,"tiny":0.0001,"user":{"active":true,"age":29.5,"name":"Ada","score":0.98,"tags":["a","b"]}}
        let nested = json!({
            "user": {"name": "Ada", "age": 29.5, "active": true, "score": 0.98, "tags": ["a", "b"]},
            "balance": 1250.0,
            "tiny": 0.0001,
            "big": 1e15,
        });
        assert_eq!(
            dump(&nested),
            "{\"balance\":1250.0,\"big\":1000000000000000.0,\"tiny\":0.0001,\
             \"user\":{\"active\":true,\"age\":29.5,\"name\":\"Ada\",\"score\":0.98,\
             \"tags\":[\"a\",\"b\"]}}"
        );

        // {"amount":12.25,"note":"héllo — 中文 \n \"quoted\" \\ back\ttab"}
        let escapes = json!({
            "note": "héllo — 中文 \n \"quoted\" \\ back\ttab",
            "amount": 12.25,
        });
        assert_eq!(
            dump(&escapes),
            "{\"amount\":12.25,\"note\":\"héllo — 中文 \\n \\\"quoted\\\" \\\\ back\\ttab\"}"
        );

        // control characters: {"ctl":"a\u0001b\u001fcd"}
        let control = json!({"ctl": "a\u{01}b\u{1f}cd"});
        assert_eq!(dump(&control), "{\"ctl\":\"a\\u0001b\\u001fcd\"}");

        // int vs float: {"f":3.0,"i":3,"neg":-7,"u":18446744073709551615}
        let numbers = json!({"i": 3, "f": 3.0, "neg": -7, "u": 18_446_744_073_709_551_615u64});
        assert_eq!(
            dump(&numbers),
            "{\"f\":3.0,\"i\":3,\"neg\":-7,\"u\":18446744073709551615}"
        );

        assert_eq!(dump(&Value::Null), "null");
        assert_eq!(dump(&json!([])), "[]");
        assert_eq!(dump(&json!({})), "{}");
        assert_eq!(dump(&json!([1, "two", false, null])), "[1,\"two\",false,null]");
    }

    /// Top-level strings render as themselves (the INSTRUCTION leg —
    /// the reference inserts the instruction text raw, unquoted).
    #[test]
    fn render_top_level_string_is_itself() {
        assert_eq!(render(&json!("Is the user gold tier?")), "Is the user gold tier?");
        assert_eq!(render(&json!({"a": 1})), "{\"a\":1}");
        assert_eq!(render(&json!(1.5)), "1.5");
        assert_eq!(render(&json!(3)), "3");
    }
}
