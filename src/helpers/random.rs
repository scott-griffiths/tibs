use super::bits::BV;
use super::validation::validate_length;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rand::rngs::{StdRng, SysRng};
use rand::{Rng, SeedableRng, TryRng};
use sha2::{Digest, Sha256};

fn process_seed(seed: &Option<Vec<u8>>) -> [u8; 32] {
    match seed {
        None => {
            let mut seed_arr = [0u8; 32];
            rand::rng().fill_bytes(&mut seed_arr);
            seed_arr
        }
        Some(seed_bytes) => {
            let mut hasher = Sha256::new();
            hasher.update(seed_bytes);
            let digest = hasher.finalize();
            let mut seed_arr = [0u8; 32];
            seed_arr.copy_from_slice(&digest);
            seed_arr
        }
    }
}

pub(crate) fn bv_from_random(length: i64, secure: bool, seed: &Option<Vec<u8>>) -> PyResult<BV> {
    let length = validate_length(length)?;
    if secure && seed.is_some() {
        return Err(PyValueError::new_err(
            "A seed cannot be used when generating secure random data.",
        ));
    }
    if length == 0 {
        return Ok(BV::new());
    }
    let num_bytes = length.div_ceil(8);
    let mut data = vec![0u8; num_bytes];
    if secure {
        SysRng
            .try_fill_bytes(&mut data)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    } else {
        let seed_arr = process_seed(seed);
        let mut rng = StdRng::from_seed(seed_arr);
        rng.fill_bytes(&mut data);
    }
    let mut bv = BV::from_vec(data);
    if bv.len() > length {
        bv.truncate(length);
    }
    Ok(bv)
}
