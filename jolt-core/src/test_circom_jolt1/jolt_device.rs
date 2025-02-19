use super::struct_fq::FqCircom;
use ark_bn254::Fr as Scalar;
use core::fmt;
use tracer::JoltDevice;

#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct JoltDeviceCircom {
    pub inputs: Vec<FqCircom>,
    pub outputs: Vec<FqCircom>,
    pub panic: FqCircom,
}

impl fmt::Debug for JoltDeviceCircom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            r#"{{
                    "inputs": {:?},
                    "outputs": {:?},
                    "panic": {:?}
            }}"#,
            self.inputs, self.outputs, self.panic
        )
    }
}

pub fn convert_from_jolt_device_to_circom(jolt_device: JoltDevice) -> JoltDeviceCircom {
    let mut inputs = Vec::new();
    for i in 0..jolt_device.inputs.len() {
        inputs.push(FqCircom(Scalar::from(jolt_device.inputs[i] as u64)));
    }
    let mut outputs = Vec::new();
    for i in 0..jolt_device.outputs.len() {
        outputs.push(FqCircom(Scalar::from(jolt_device.outputs[i] as u64)));
    }
    JoltDeviceCircom {
        inputs,
        outputs,
        panic: FqCircom(Scalar::from(jolt_device.panic as u64)),
    }
}
