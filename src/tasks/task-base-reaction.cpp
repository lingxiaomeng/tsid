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
#include <pinocchio/algorithm/joint-configuration.hpp>

namespace tsid {
namespace tasks {
using namespace math;
using namespace trajectories;

TaskBaseReaction::TaskBaseReaction(const std::string& name, RobotWrapper& robot)
    : TaskMotion(name, robot),
      m_base_dim(robot.nv() - robot.na()),
      m_arm_dim(robot.na()),
      m_posture_ref(robot.nq_actuated(), robot.na()),
      m_use_projector(false),
      m_constraint(name, robot.nv() - robot.na(), robot.nv()) {
  m_reference.setZero(m_base_dim);
  m_Kp.setZero(m_arm_dim);
  m_Kd.setZero(m_arm_dim);
  m_p_error.setZero(m_arm_dim);
  m_v_error.setZero(m_arm_dim);
  m_p.setZero(robot.nq_actuated());
  m_v.setZero(m_arm_dim);
  m_a_des.setZero(m_arm_dim);
  m_ref_q_augmented = pinocchio::neutral(robot.model());
  m_projector.setIdentity(robot.na(), robot.na());
  m_projected_row.setZero(1, robot.na());
  Vector mask = Vector::Ones(m_base_dim);
  setMask(mask);
}

void TaskBaseReaction::setReference(ConstRefVector reference) {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(
      reference.size() == m_base_dim,
      "The size of the base reaction reference needs to equal " +
          std::to_string(m_base_dim));
  m_reference = reference;
}

void TaskBaseReaction::setPostureReference(const TrajectorySample& ref) {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(
      ref.getValue().size() == m_robot.nq_actuated(),
      "The size of the posture reference value vector needs to equal " +
          std::to_string(m_robot.nq_actuated()));
  PINOCCHIO_CHECK_INPUT_ARGUMENT(
      ref.getDerivative().size() == m_robot.na(),
      "The size of the posture reference derivative vector needs to equal " +
          std::to_string(m_robot.na()));
  PINOCCHIO_CHECK_INPUT_ARGUMENT(
      ref.getSecondDerivative().size() == m_robot.na(),
      "The size of the posture reference second derivative vector needs to "
      "equal " +
          std::to_string(m_robot.na()));
  m_posture_ref = ref;
}

const TaskBaseReaction::TrajectorySample& TaskBaseReaction::getReference()
    const {
  return m_posture_ref;
}

const Vector& TaskBaseReaction::getDesiredAcceleration() const {
  return m_a_des;
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
  m_constraint.resize((unsigned int)dim, m_robot.nv());
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

int TaskBaseReaction::dim() const { return (int)m_mask.sum(); }

const Vector& TaskBaseReaction::Kp() { return m_Kp; }

const Vector& TaskBaseReaction::Kd() { return m_Kd; }

void TaskBaseReaction::Kp(ConstRefVector Kp) {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(Kp.size() == m_robot.na(),
                                 "The size of the Kp vector needs to equal " +
                                     std::to_string(m_robot.na()));
  m_Kp = Kp;
}

void TaskBaseReaction::Kd(ConstRefVector Kd) {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(Kd.size() == m_robot.na(),
                                 "The size of the Kd vector needs to equal " +
                                     std::to_string(m_robot.na()));
  m_Kd = Kd;
}

const ConstraintBase& TaskBaseReaction::getConstraint() const {
  return m_constraint;
}

const ConstraintBase& TaskBaseReaction::compute(const double, ConstRefVector q,
                                                ConstRefVector v, Data& data) {
  const Matrix& M = m_robot.mass(data);
  m_ref_q_augmented.tail(m_robot.nq_actuated()) = m_posture_ref.getValue();

  m_p = q.tail(m_robot.nq_actuated());
  m_p_error = pinocchio::difference(m_robot.model(), m_ref_q_augmented, q)
                  .tail(m_robot.na());
  m_v = v.tail(m_robot.na());
  m_v_error = m_v - m_posture_ref.getDerivative();
  m_a_des = -m_Kp.cwiseProduct(m_p_error) - m_Kd.cwiseProduct(m_v_error) +
            m_posture_ref.getSecondDerivative();

  m_constraint.matrix().setZero();
  for (unsigned int i = 0; i < m_activeAxes.size(); i++) {
    const unsigned int axis = m_activeAxes(i);

    if (m_use_projector) {
      m_projected_row.noalias() =
          M.row(axis).segment(m_base_dim, m_arm_dim) * m_projector;
      m_constraint.matrix().row(i).segment(m_base_dim, m_arm_dim) =
          m_projected_row;
      m_constraint.vector()(i) =
          m_reference(axis) + (m_projected_row * m_a_des)(0);
    } else {
      m_constraint.matrix().row(i).segment(m_base_dim, m_arm_dim) =
          M.row(axis).segment(m_base_dim, m_arm_dim);
      m_constraint.vector()(i) =
          m_reference(axis) +
          M.row(axis).segment(m_base_dim, m_arm_dim).dot(m_a_des);
    }
  }
  return m_constraint;
}

}  // namespace tasks
}  // namespace tsid
