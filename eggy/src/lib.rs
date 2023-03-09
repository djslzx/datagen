use pyo3::prelude::*;
mod lsystem;

#[pyfunction]
fn simplify_lsystem(s: &str) -> PyResult<String> {
    Ok(lsystem::simplify(s))
}

#[pymodule]
fn eggy(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simplify_lsystem, m)?)?;
    Ok(())
}
