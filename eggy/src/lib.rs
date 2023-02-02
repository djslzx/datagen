use pyo3::prelude::*;
mod simpl;

#[pyfunction]
fn simplify(s: &str) -> PyResult<String> {
    Ok(simpl::simplify(s))
}

#[pymodule]
fn eggy(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simplify, m)?)?;
    Ok(())
}
