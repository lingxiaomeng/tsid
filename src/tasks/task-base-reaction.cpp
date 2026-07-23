//
// Copyright (c) 2017 CNRS
//
// This file is part of tsid
// tsid is free software: you can redistribute it
// and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation, either version
// 3 of the License, or (at your option) any later version.
// tsid is distributed in the hope that it will be
// useful, but WITHOUT ANY WARRANTY; without even the implied warranty
// of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// General Lesser Public License for more details. You should have
// received a copy of the GNU Lesser General Public License along with
// tsid If not, see
// <http://www.gnu.org/licenses/>.
//

#include <tsid/tasks/task-base-reaction.hpp>
#include "tsid/robots/robot-wrapper.hpp"

#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>

#include <Eigen/Geometry>

#include <chrono>
#include <cmath>

namespace tsid {
namespace tasks {
using namespace math;
TaskBaseReaction::TaskBaseReaction(const std::string& name, RobotWrapper& robot)
    : TaskBaseReaction(name, robot, robot.model()) {}

TaskBaseReaction::TaskBaseReaction(
    const std::string& name, RobotWrapper& robot,
    const pinocchio::Model& free_flyer_model)
    : TaskMotion(name, robot),
      m_base_dim(6),
      m_arm_dim(robot.na()),
      m_model_ff(free_flyer_model),
      m_data_ff(m_model_ff),
      m_input_base_joint(0),
      m_ff_base_joint(0),
      m_computation_time_us(0.0),
      m_use_projector(false),
      m_reaction_weight_sqrt(0.0),
      m_gravity_weight_sqrt(0.0),
      m_gravity_prediction_horizon(0.1),
      m_constraint(name, 6, robot.nv()) {
  buildJointIndexMap();

  m_reference.setZero(m_base_dim);
  m_q_ff = pinocchio::neutral(m_model_ff);
  m_v_ff.setZero(m_model_ff.nv);
  m_a_ff.setZero(m_model_ff.nv);
  m_base_acceleration.setZero(6);
  m_M_BB.setZero(6, 6);
  m_M_Bq.setZero(6, m_arm_dim);
  m_h_B.setZero(6);
  m_h_com.setZero(6);
  m_tau_rnea.setZero(m_model_ff.nv);
  m_affine_check.setZero(6);
  m_gravity_partial_dq.setZero(m_model_ff.nv, m_model_ff.nv);
  m_gravity_jacobian_mount.setZero(6, m_arm_dim);
  m_gravity_jacobian_com.setZero(6, m_arm_dim);
  m_projected_gravity_jacobian_com.setZero(6, m_arm_dim);
  m_gravity_reaction_mount.setZero(6);
  m_gravity_reaction_com.setZero(6);
  m_arm_velocity.setZero(m_arm_dim);
  m_gravity_roll_pitch_torque.setZero();
  m_gravity_prediction_offset.setZero();
  m_predicted_gravity_roll_pitch_torque.setZero();
  m_wrench_mount_to_com.setIdentity();
  m_base_matrix_com.setZero();
  m_axis_scale.setOnes();
  m_G_mount.setZero(6, m_arm_dim);
  m_w0_mount.setZero(6);
  m_G_com.setZero(6, m_arm_dim);
  m_w0_com.setZero(6);
  m_G_roll_pitch.setZero(2, m_arm_dim);
  m_b_roll_pitch.setZero();
  m_reaction_mount.setZero(6);
  m_reaction_com.setZero(6);
  m_roll_pitch_torque.setZero();
  m_projector.setIdentity(robot.na(), robot.na());
  m_projected_G_com.setZero(6, robot.na());
  Vector mask = Vector::Ones(m_base_dim);
  setMask(mask);
}

void TaskBaseReaction::buildJointIndexMap() {
  for (pinocchio::JointIndex joint = 1;
       joint < static_cast<pinocchio::JointIndex>(m_model_ff.njoints); ++joint) {
    if (m_model_ff.joints[joint].nq() == 7 &&
        m_model_ff.joints[joint].nv() == 6) {
      m_ff_base_joint = joint;
      break;
    }
  }
  PINOCCHIO_CHECK_INPUT_ARGUMENT(
      m_ff_base_joint != 0,
      "The reaction model must contain a JointModelFreeFlyer root joint");

  const pinocchio::Model& input_model = m_robot.model();
  for (pinocchio::JointIndex joint = 1;
       joint < static_cast<pinocchio::JointIndex>(input_model.njoints); ++joint) {
    if (input_model.joints[joint].nq() == 7 &&
        input_model.joints[joint].nv() == 6) {
      m_input_base_joint = joint;
      break;
    }
  }

  int actuated_idx_v = 0;
  for (pinocchio::JointIndex input_joint = 1;
       input_joint < static_cast<pinocchio::JointIndex>(input_model.njoints);
       ++input_joint) {
    if (input_joint == m_input_base_joint ||
        input_model.joints[input_joint].nv() == 0) {
      continue;
    }

    const std::string& joint_name = input_model.names[input_joint];
    PINOCCHIO_CHECK_INPUT_ARGUMENT(
        m_model_ff.existJointName(joint_name),
        "Joint " + joint_name + " is missing from the free-flyer model");
    const pinocchio::JointIndex ff_joint = m_model_ff.getJointId(joint_name);
    const int nq = input_model.joints[input_joint].nq();
    const int nv = input_model.joints[input_joint].nv();
    PINOCCHIO_CHECK_INPUT_ARGUMENT(
        m_model_ff.joints[ff_joint].nq() == nq &&
            m_model_ff.joints[ff_joint].nv() == nv,
        "Joint " + joint_name +
            " has different dimensions in the fixed and free-flyer models");

    m_joint_index_map.push_back(
        {input_joint,
         ff_joint,
         input_model.joints[input_joint].idx_q(),
         m_model_ff.joints[ff_joint].idx_q(),
         nq,
         input_model.joints[input_joint].idx_v(),
         m_model_ff.joints[ff_joint].idx_v(),
         nv,
         actuated_idx_v});
    actuated_idx_v += nv;
  }
  PINOCCHIO_CHECK_INPUT_ARGUMENT(
      actuated_idx_v == static_cast<int>(m_arm_dim),
      "The fixed-to-free-flyer joint map does not cover all actuated joints");
}

void TaskBaseReaction::setReference(ConstRefVector reference) {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(
      reference.size() == m_base_dim,
      "The size of the base reaction reference needs to equal " +
          std::to_string(m_base_dim));
  m_reference = reference;
}

void TaskBaseReaction::setBaseKinematics(
    const pinocchio::SE3& placement_world_base,
    const pinocchio::Motion& velocity_body,
    const pinocchio::Motion& classical_acceleration_body) {
  const int idx_q = m_model_ff.joints[m_ff_base_joint].idx_q();
  const int idx_v = m_model_ff.joints[m_ff_base_joint].idx_v();
  const Eigen::Quaterniond quaternion(placement_world_base.rotation());
  m_q_ff.segment<3>(idx_q) = placement_world_base.translation();
  m_q_ff.segment<4>(idx_q + 3) = quaternion.normalized().coeffs();
  m_v_ff.segment<3>(idx_v) = velocity_body.linear();
  m_v_ff.segment<3>(idx_v + 3) = velocity_body.angular();
  m_base_acceleration.head<3>() =
      classical_acceleration_body.linear() -
      velocity_body.angular().cross(velocity_body.linear());
  m_base_acceleration.tail<3>() = classical_acceleration_body.angular();
}

void TaskBaseReaction::setBaseAcceleration(
    ConstRefVector spatial_acceleration_body) {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(
      spatial_acceleration_body.size() == 6,
      "The base spatial acceleration must have six elements");
  m_base_acceleration = spatial_acceleration_body;
}

void TaskBaseReaction::setMountToComTransform(
    ConstRefMatrix rotation_com_mount,
    ConstRefVector position_com_to_mount_com) {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(
      rotation_com_mount.rows() == 3 && rotation_com_mount.cols() == 3,
      "The mount-to-COM rotation must be 3 x 3");
  PINOCCHIO_CHECK_INPUT_ARGUMENT(
      position_com_to_mount_com.size() == 3,
      "The COM-to-mount position must have three elements");

  Eigen::Matrix3d skew;
  const Eigen::Vector3d r = position_com_to_mount_com;
  skew << 0.0, -r.z(), r.y(), r.z(), 0.0, -r.x(), -r.y(), r.x(), 0.0;
  m_wrench_mount_to_com.setZero();
  m_wrench_mount_to_com.topLeftCorner<3, 3>() = rotation_com_mount;
  m_wrench_mount_to_com.bottomRightCorner<3, 3>() = rotation_com_mount;
  m_wrench_mount_to_com.bottomLeftCorner<3, 3>() =
      skew * rotation_com_mount;
}

void TaskBaseReaction::setRollPitchTorqueScale(double roll_scale,
                                               double pitch_scale) {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(
      roll_scale > 0.0 && pitch_scale > 0.0,
      "The roll and pitch torque scales must be positive");
  m_axis_scale(3) = roll_scale;
  m_axis_scale(4) = pitch_scale;
}

void TaskBaseReaction::setObjectiveWeights(double reaction_weight,
                                           double gravity_weight) {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(
      reaction_weight >= 0.0 && gravity_weight >= 0.0,
      "The reaction and gravity objective weights must be non-negative");
  m_reaction_weight_sqrt = std::sqrt(reaction_weight);
  m_gravity_weight_sqrt = std::sqrt(gravity_weight);
}

void TaskBaseReaction::setGravityPredictionHorizon(
    double prediction_horizon) {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(
      prediction_horizon > 0.0,
      "The gravity prediction horizon must be positive");
  m_gravity_prediction_horizon = prediction_horizon;
}

void TaskBaseReaction::setMask(ConstRefVector mask) {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(
      mask.size() == m_base_dim,
      "The size of the base reaction mask needs to equal " +
          std::to_string(m_base_dim));
  m_mask = mask;
  const Vector::Index dim = static_cast<Vector::Index>(mask.sum());
  m_activeAxes.resize(dim);
  unsigned int j = 0;
  for (unsigned int i = 0; i < mask.size(); i++) {
    if (mask(i) != 0.0) {
      PINOCCHIO_CHECK_INPUT_ARGUMENT(
          mask(i) == 1.0,
          "Valid base reaction mask values are either 0.0 or 1.0 received: " +
              std::to_string(mask(i)));
      m_activeAxes(j) = i;
      j++;
    }
  }
  m_constraint.resize((unsigned int)(2 * dim), m_robot.nv());
}

void TaskBaseReaction::setProjector(ConstRefMatrix projector) {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(
      projector.rows() == m_arm_dim && projector.cols() == m_arm_dim,
      "The base reaction projector needs to be " + std::to_string(m_arm_dim) +
          " x " + std::to_string(m_arm_dim));
  m_use_projector = true;
  m_projector = projector;
}

void TaskBaseReaction::disableProjector() { m_use_projector = false; }

int TaskBaseReaction::dim() const { return 2 * (int)m_mask.sum(); }

const ConstraintBase& TaskBaseReaction::getConstraint() const {
  return m_constraint;
}

void TaskBaseReaction::updateFreeFlyerState(ConstRefVector q,
                                             ConstRefVector v) {
  if (m_input_base_joint != 0) {
    const pinocchio::JointModel& input_base =
        m_robot.model().joints[m_input_base_joint];
    const pinocchio::JointModel& ff_base =
        m_model_ff.joints[m_ff_base_joint];
    m_q_ff.segment(ff_base.idx_q(), 7) = q.segment(input_base.idx_q(), 7);
    Eigen::Map<Eigen::Quaterniond> quaternion(
        m_q_ff.data() + ff_base.idx_q() + 3);
    quaternion.normalize();
    m_v_ff.segment(ff_base.idx_v(), 6) = v.segment(input_base.idx_v(), 6);
  }

  for (const JointIndexMap& map : m_joint_index_map) {
    m_q_ff.segment(map.free_flyer_idx_q, map.nq) =
        q.segment(map.input_idx_q, map.nq);
    m_v_ff.segment(map.free_flyer_idx_v, map.nv) =
        v.segment(map.input_idx_v, map.nv);
  }
}

const ConstraintBase& TaskBaseReaction::compute(const double, ConstRefVector q,
                                                ConstRefVector v, Data&) {
  const auto start = std::chrono::steady_clock::now();
  if (!q.allFinite() || !v.allFinite() || !m_base_acceleration.allFinite()) {
    m_constraint.matrix().setZero();
    m_constraint.vector().setZero();
    m_G_mount.setZero();
    m_w0_mount.setZero();
    m_G_com.setZero();
    m_w0_com.setZero();
    m_G_roll_pitch.setZero();
    m_b_roll_pitch.setZero();
    m_reaction_mount.setZero();
    m_reaction_com.setZero();
    m_roll_pitch_torque.setZero();
    m_gravity_jacobian_mount.setZero();
    m_gravity_jacobian_com.setZero();
    m_projected_gravity_jacobian_com.setZero();
    m_gravity_reaction_mount.setZero();
    m_gravity_reaction_com.setZero();
    m_gravity_roll_pitch_torque.setZero();
    m_gravity_prediction_offset.setZero();
    m_predicted_gravity_roll_pitch_torque.setZero();
    m_computation_time_us =
        std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - start)
            .count();
    return m_constraint;
  }
  updateFreeFlyerState(q, v);

  pinocchio::crba(m_model_ff, m_data_ff, m_q_ff);
  m_data_ff.M.triangularView<Eigen::StrictlyLower>() =
      m_data_ff.M.transpose().triangularView<Eigen::StrictlyLower>();
  pinocchio::nonLinearEffects(m_model_ff, m_data_ff, m_q_ff, m_v_ff);

  const int base_idx_v = m_model_ff.joints[m_ff_base_joint].idx_v();
  m_M_BB = m_data_ff.M.block(base_idx_v, base_idx_v, 6, 6);
  m_M_Bq.setZero();
  for (const JointIndexMap& map : m_joint_index_map) {
    m_M_Bq.middleCols(map.actuated_idx_v, map.nv) =
        m_data_ff.M.block(base_idx_v, map.free_flyer_idx_v, 6, map.nv);
  }
  m_h_B = m_data_ff.nle.segment(base_idx_v, 6);
  m_h_com.noalias() = m_wrench_mount_to_com * m_h_B;
  m_base_matrix_com.noalias() = -m_wrench_mount_to_com * m_M_BB;

  m_G_mount.noalias() = -m_M_Bq;
  m_w0_mount.noalias() = -(m_M_BB * m_base_acceleration + m_h_B);
  m_G_com.noalias() = m_wrench_mount_to_com * m_G_mount;
  m_w0_com.noalias() = m_wrench_mount_to_com * m_w0_mount;
  m_G_roll_pitch = m_G_com.middleRows(3, 2);
  m_b_roll_pitch = m_w0_com.segment<2>(3);
  m_reaction_mount = m_w0_mount;
  m_reaction_com = m_w0_com;
  m_roll_pitch_torque = m_b_roll_pitch;

  if (m_use_projector) {
    m_projected_G_com.noalias() = m_G_com * m_projector;
  } else {
    m_projected_G_com = m_G_com;
  }

  if (m_gravity_weight_sqrt > 0.0) {
    m_gravity_partial_dq.setZero();
    pinocchio::computeGeneralizedGravityDerivatives(
        m_model_ff, m_data_ff, m_q_ff, m_gravity_partial_dq);
  } else {
    pinocchio::computeGeneralizedGravity(m_model_ff, m_data_ff, m_q_ff);
  }
  m_gravity_reaction_mount = -m_data_ff.g.segment(base_idx_v, 6);
  m_gravity_reaction_com.noalias() =
      m_wrench_mount_to_com * m_gravity_reaction_mount;
  m_gravity_jacobian_mount.setZero();
  m_arm_velocity.setZero();
  for (const JointIndexMap& map : m_joint_index_map) {
    if (m_gravity_weight_sqrt > 0.0) {
      m_gravity_jacobian_mount.middleCols(map.actuated_idx_v, map.nv) =
          -m_gravity_partial_dq.block(
              base_idx_v, map.free_flyer_idx_v, 6, map.nv);
    }
    m_arm_velocity.segment(map.actuated_idx_v, map.nv) =
        m_v_ff.segment(map.free_flyer_idx_v, map.nv);
  }
  m_gravity_jacobian_com.noalias() =
      m_wrench_mount_to_com * m_gravity_jacobian_mount;
  if (m_use_projector) {
    m_projected_gravity_jacobian_com.noalias() =
        m_gravity_jacobian_com * m_projector;
  } else {
    m_projected_gravity_jacobian_com = m_gravity_jacobian_com;
  }
  m_gravity_roll_pitch_torque =
      m_gravity_reaction_com.segment<2>(3);
  m_gravity_prediction_offset.noalias() =
      m_gravity_roll_pitch_torque +
      m_gravity_jacobian_com.middleRows(3, 2) * m_arm_velocity *
          m_gravity_prediction_horizon;
  m_predicted_gravity_roll_pitch_torque = m_gravity_prediction_offset;

  m_constraint.matrix().setZero();
  m_constraint.vector().setZero();
  const unsigned int gravity_row_offset =
      static_cast<unsigned int>(m_activeAxes.size());
  const double acceleration_prediction_scale =
      0.5 * m_gravity_prediction_horizon * m_gravity_prediction_horizon;
  for (unsigned int i = 0; i < m_activeAxes.size(); i++) {
    const unsigned int axis = m_activeAxes(i);
    const double scale = m_axis_scale(axis);

    if (m_input_base_joint != 0) {
      const int input_base_idx_v =
          m_robot.model().joints[m_input_base_joint].idx_v();
      m_constraint.matrix().row(i).segment(input_base_idx_v, 6) =
          m_reaction_weight_sqrt * m_base_matrix_com.row(axis) / scale;
    }

    for (const JointIndexMap& map : m_joint_index_map) {
      m_constraint.matrix().row(i).segment(map.input_idx_v, map.nv) =
          m_reaction_weight_sqrt * m_projected_G_com.row(axis)
              .segment(map.actuated_idx_v, map.nv) /
          scale;
      m_constraint.matrix()
          .row(gravity_row_offset + i)
          .segment(map.input_idx_v, map.nv) =
          m_gravity_weight_sqrt * acceleration_prediction_scale *
          m_projected_gravity_jacobian_com.row(axis)
              .segment(map.actuated_idx_v, map.nv) /
          scale;
    }

    double rhs = m_reference(axis) + m_h_com(axis);
    if (m_input_base_joint == 0) {
      rhs = m_reference(axis) - m_w0_com(axis);
    }
    m_constraint.vector()(i) = m_reaction_weight_sqrt * rhs / scale;
    const double gravity_offset =
        m_gravity_reaction_com(axis) +
        m_gravity_jacobian_com.row(axis).dot(m_arm_velocity) *
            m_gravity_prediction_horizon;
    m_constraint.vector()(gravity_row_offset + i) =
        -m_gravity_weight_sqrt * gravity_offset / scale;
  }

  m_computation_time_us =
      std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() -
                                                start)
          .count();
  return m_constraint;
}

void TaskBaseReaction::updateReaction(ConstRefVector arm_acceleration) {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(
      arm_acceleration.size() == static_cast<int>(m_arm_dim),
      "The arm acceleration size must equal the number of actuated joints");
  m_reaction_mount.noalias() = m_G_mount * arm_acceleration + m_w0_mount;
  m_reaction_com.noalias() = m_G_com * arm_acceleration + m_w0_com;
  m_roll_pitch_torque = m_reaction_com.segment<2>(3);
  m_predicted_gravity_roll_pitch_torque.noalias() =
      m_gravity_prediction_offset +
      0.5 * m_gravity_prediction_horizon * m_gravity_prediction_horizon *
          m_gravity_jacobian_com.middleRows(3, 2) * arm_acceleration;
}

double TaskBaseReaction::validateAffineModel(ConstRefVector arm_acceleration) {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(
      arm_acceleration.size() == static_cast<int>(m_arm_dim),
      "The arm acceleration size must equal the number of actuated joints");
  m_a_ff.setZero();
  const int base_idx_v = m_model_ff.joints[m_ff_base_joint].idx_v();
  m_a_ff.segment(base_idx_v, 6) = m_base_acceleration;
  for (const JointIndexMap& map : m_joint_index_map) {
    m_a_ff.segment(map.free_flyer_idx_v, map.nv) =
        arm_acceleration.segment(map.actuated_idx_v, map.nv);
  }
  m_tau_rnea =
      pinocchio::rnea(m_model_ff, m_data_ff, m_q_ff, m_v_ff, m_a_ff);
  m_affine_check.noalias() =
      m_G_mount * arm_acceleration + m_w0_mount;
  return (-m_tau_rnea.segment(base_idx_v, 6) - m_affine_check)
      .lpNorm<Eigen::Infinity>();
}

const Vector& TaskBaseReaction::reactionAtMount() const {
  return m_reaction_mount;
}

const Vector& TaskBaseReaction::reactionAtCom() const { return m_reaction_com; }

const Eigen::Vector2d& TaskBaseReaction::rollPitchTorque() const {
  return m_roll_pitch_torque;
}

const Matrix& TaskBaseReaction::reactionMatrixAtMount() const {
  return m_G_mount;
}

const Vector& TaskBaseReaction::reactionOffsetAtMount() const {
  return m_w0_mount;
}

const Matrix& TaskBaseReaction::reactionMatrixAtCom() const { return m_G_com; }

const Vector& TaskBaseReaction::reactionOffsetAtCom() const { return m_w0_com; }

const Eigen::Matrix<double, 2, Eigen::Dynamic>&
TaskBaseReaction::rollPitchMatrix() const {
  return m_G_roll_pitch;
}

const Eigen::Vector2d& TaskBaseReaction::rollPitchOffset() const {
  return m_b_roll_pitch;
}

const Eigen::Vector2d& TaskBaseReaction::gravityRollPitchTorque() const {
  return m_gravity_roll_pitch_torque;
}

const Eigen::Vector2d&
TaskBaseReaction::predictedGravityRollPitchTorque() const {
  return m_predicted_gravity_roll_pitch_torque;
}

double TaskBaseReaction::computationTimeUs() const {
  return m_computation_time_us;
}

}  // namespace tasks
}  // namespace tsid
