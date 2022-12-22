pub mod lib;
use crate::lib::relaxed_ik;
use crate::lib::utils_rust::subscriber_utils::JointsAnglesSubscriber;
use crate::lib::utils_rust::subscriber_utils::*;
use nalgebra::{Quaternion, UnitQuaternion, Vector3};
use rosrust;
use rosrust_msg::sensor_msgs::JointState;
use std::sync::{Arc, Mutex};

mod msg {
    rosrust::rosmsg_include!(relaxed_ik / EEPoseGoals, relaxed_ik / JointAngles);
}

fn main() {
    rosrust::init("TEST_relaxed_ik_ALBAN");

    println!("START TESTING");

    let mut r = relaxed_ik::RelaxedIK::from_loaded(1);

    // let arc = Arc::new(Mutex::new(JointsAnglesSubscriber::new()));
    // let arc2 = arc.clone();
    let subscriber = rosrust::subscribe(
        "joint_states",
        3,
        move |v: rosrust_msg::sensor_msgs::JointState| {
            // let mut g = arc2.lock().unwrap();
            // g.joints_angles = Vec::new();
            // let num_joints = v.position.len();
            // for i in 0..num_joints {
            // g.joints_angles.push(v.position[i]);
            // }
            println!("recevied {}", v.position)
        },
    )
    .unwrap();

    rosrust::spin();

    // let rate1 = rosrust::rate(100.);
    // while arc.lock().unwrap().joints_angles.is_empty() {
    //     println!("waiting");
    //     rate1.sleep();
    // }

    // let rate = rosrust::rate(1.);
    // while rosrust::is_ok() {
    //     let mut thing = &arc.lock().unwrap();

    //     println!("AAAAAAAAAAAAAAAAAAAAAAAAA{:?}", thing.joints_angles);

    //     r.vars.robot.arms[0].get_frames_immutable(&thing.joints_angles);
    //     rate.sleep();
    // }
}
