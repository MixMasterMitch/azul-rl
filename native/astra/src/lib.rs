mod batch;
mod eval;
mod game;
#[cfg(test)]
mod game_tests;
mod search;
mod state;
#[cfg(test)]
mod tests;
mod tree;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use state::State;

fn parse(snapshot: &[i32]) -> PyResult<State> {
    State::parse(snapshot).map_err(PyValueError::new_err)
}

#[pyfunction]
fn _binding_roundtrip(snapshot: Vec<i32>) -> Vec<i32> {
    snapshot
}

#[pyfunction]
fn legal_actions(snapshot: Vec<i32>) -> PyResult<Vec<u16>> {
    Ok(parse(&snapshot)?.legal())
}

#[pyfunction]
#[pyo3(signature = (snapshot, action, resolve=true))]
fn transition(snapshot: Vec<i32>, action: u16, resolve: bool) -> PyResult<Vec<i32>> {
    let s = parse(&snapshot)?;
    if !s.legal().contains(&action) {
        return Err(PyValueError::new_err("Illegal action"));
    }
    Ok(s.apply(action, resolve).pack())
}

#[pyfunction]
fn resolve_round(snapshot: Vec<i32>) -> PyResult<Vec<i32>> {
    let mut s = parse(&snapshot)?;
    if !s.empty() {
        return Err(PyValueError::new_err("Round still has tiles"));
    }
    s.resolve();
    Ok(s.pack())
}

#[pyfunction]
#[pyo3(signature = (snapshot, nodes=4000, time_ms=2000, depth=32, width=12, weights=None, seed=0, rollout=false, center_nodes=0))]
#[allow(clippy::too_many_arguments)] // Keyword arguments are the Python-facing configuration API.
fn analyze(
    py: Python<'_>,
    snapshot: Vec<i32>,
    nodes: u64,
    time_ms: u64,
    depth: u8,
    width: usize,
    weights: Option<Vec<f64>>,
    seed: u64,
    rollout: bool,
    center_nodes: u64,
) -> PyResult<Py<PyDict>> {
    let s = parse(&snapshot)?;
    if s.legal().is_empty() {
        return Err(PyValueError::new_err("No legal action in this state"));
    }
    if nodes == 0
        || nodes > 100_000_000
        || center_nodes > 100_000_000
        || !(1..=2000).contains(&time_ms)
        || depth > 64
        || width > 300
    {
        return Err(PyValueError::new_err("Invalid search limits"));
    }
    let w = weights.unwrap_or_else(|| eval::DEFAULT_WEIGHTS.to_vec());
    if w.len() != 11
        || w.iter().any(|x| !x.is_finite() || *x < 0.0 || *x > 100.0)
        || w[7] == 0.0
        || w[8] > 1.0
        || w[9] > 1.0
        || w[10] > 1.0
    {
        return Err(PyValueError::new_err(
            "Expected eleven finite nonnegative weights; urgency must be positive and column_prior/bonus_link/field_pressure at most one",
        ));
    }
    let cfg = search::Config {
        nodes,
        center_nodes,
        time_ms,
        depth,
        width,
        weights: w.try_into().unwrap(),
        seed,
        rollout,
    };
    let r = py.detach(|| search::run(&s, cfg));
    let d = PyDict::new(py);
    d.set_item("action", r.action)?;
    d.set_item("nodes", r.nodes)?;
    d.set_item("evaluations", r.evaluations)?;
    d.set_item("tt_hits", r.tt_hits)?;
    d.set_item("evaluation_cache_hits", r.evaluation_cache_hits)?;
    d.set_item("depth", r.depth)?;
    d.set_item("elapsed_s", r.elapsed)?;
    d.set_item("cutoff_reason", r.reason)?;
    d.set_item("principal_variation", r.pv)?;
    d.set_item("values", r.value.to_vec())?;
    d.set_item(
        "components",
        r.components[..s.n]
            .iter()
            .map(|v| v.to_vec())
            .collect::<Vec<_>>(),
    )?;
    d.set_item("component_names", eval::COMPONENTS.to_vec())?;
    d.set_item("solved", r.solved)?;
    Ok(d.unbind())
}

#[pymodule]
fn azul_astra(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("SNAPSHOT_VERSION", 1)?;
    m.add("CONFIG_VERSION", 3)?;
    m.add("STATE_BYTES", std::mem::size_of::<State>())?;
    m.add_function(wrap_pyfunction!(_binding_roundtrip, m)?)?;
    m.add("FULL_SNAPSHOT_VERSION", game::FULL_VERSION)?;
    m.add_class::<batch::BatchEngine>()?;
    m.add_function(wrap_pyfunction!(legal_actions, m)?)?;
    m.add_function(wrap_pyfunction!(transition, m)?)?;
    m.add_function(wrap_pyfunction!(resolve_round, m)?)?;
    m.add_function(wrap_pyfunction!(analyze, m)?)?;
    m.add_function(wrap_pyfunction!(tree::gumbel_tree, m)?)?;
    Ok(())
}
