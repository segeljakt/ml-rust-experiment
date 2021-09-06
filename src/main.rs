use pyo3::prelude::*;

const CODE: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/", "python/dnn.py"));

fn main() -> PyResult<()> {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let sys = py.import("sys")?;
        sys.setattr("argv", vec!["dnn"].into_py(py))?;
        let dnn = PyModule::from_code(py, CODE, "dnn.py", "dnn")?;
        let model = dnn.getattr("load")?.call0()?;
        let ys = vec![
            predict(py, dnn, model, 1, 85, 66, 29, 0, 26.6, 0.351, 31)?,
            predict(py, dnn, model, 8, 183, 64, 0, 0, 23.3, 0.672, 32)?,
            predict(py, dnn, model, 1, 89, 66, 23, 94, 28.1, 0.167, 21)?,
            predict(py, dnn, model, 0, 137, 40, 35, 168, 43.1, 2.288, 33)?,
            predict(py, dnn, model, 5, 116, 74, 0, 0, 25.6, 0.201, 30)?,
            predict(py, dnn, model, 3, 78, 50, 32, 88, 31.0, 0.248, 26)?,
            predict(py, dnn, model, 10, 115, 0, 0, 0, 35.3, 0.134, 29)?,
            predict(py, dnn, model, 2, 197, 70, 45, 543, 30.5, 0.158, 53)?,
            predict(py, dnn, model, 8, 125, 96, 0, 0, 0.0, 0.232, 54)?,
            predict(py, dnn, model, 4, 110, 92, 0, 0, 37.6, 0.191, 30)?,
            predict(py, dnn, model, 10, 168, 74, 0, 0, 38.0, 0.537, 34)?,
            predict(py, dnn, model, 10, 139, 80, 0, 0, 27.1, 1.441, 57)?,
            predict(py, dnn, model, 1, 189, 60, 23, 846, 30.1, 0.398, 59)?,
            predict(py, dnn, model, 5, 166, 72, 19, 175, 25.8, 0.587, 51)?,
            predict(py, dnn, model, 7, 100, 0, 0, 0, 30.0, 0.484, 32)?,
            predict(py, dnn, model, 0, 118, 84, 47, 230, 45.8, 0.551, 31)?,
            predict(py, dnn, model, 7, 107, 74, 0, 0, 29.6, 0.254, 31)?,
        ];
        for y in ys {
            println!("{}", y);
        }
        Ok(())
    })
}

fn predict(
    py: Python,
    dnn: &PyModule,
    model: &pyo3::types::PyAny,
    number_of_times_pregnant: i32,
    plasma_glucose_concentration: i32,
    diastolic_blood_pressure: i32,
    triceps_skin_fold_thickness: i32,
    serum_insulin: i32,
    body_mass_index: f32,
    diabetes_pedigree_function: f32,
    age: i32,
) -> PyResult<bool> {
    let features = vec![
        number_of_times_pregnant.into_py(py),
        plasma_glucose_concentration.into_py(py),
        diastolic_blood_pressure.into_py(py),
        triceps_skin_fold_thickness.into_py(py),
        serum_insulin.into_py(py),
        body_mass_index.into_py(py),
        diabetes_pedigree_function.into_py(py),
        age.into_py(py),
    ]
    .into_py(py);
    let args = (model, features);
    dnn.getattr("predict")?.call1(args)?.extract()
}
