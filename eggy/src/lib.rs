use pyo3::prelude::*;
mod lsystem;
mod regex;

#[pyfunction]
fn simplify_lsystem(s: &str) -> PyResult<String> {
    Ok(lsystem::simplify(s))
}

#[pyfunction]
fn simplify_regex(s: &str) -> PyResult<String> {
    Ok(regex::simplify(s))
}

#[pymodule]
fn eggy(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simplify_lsystem, m)?)?;
    m.add_function(wrap_pyfunction!(simplify_regex, m)?)?;
    Ok(())
}
