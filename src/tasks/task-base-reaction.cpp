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

namespace tsid {
namespace tasks {
using namespace math;

TaskBaseReaction::TaskBaseReaction(const std::string& name, RobotWrapper& robot)
    : TaskMotion(name, robot),
      m_base_dim(robot.nv() - robot.na()),
      m_arm_dim(robot.na()),
      m_use_projector(false),
      m_constraint(name, robot.nv() - robot.na(), robot.nv()) {
  m_reference.setZero(m_base_dim);
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

const ConstraintBase& TaskBaseReaction::getConstraint() const {
  return m_constraint;
}

const ConstraintBase& TaskBaseReaction::compute(const double, ConstRefVector,
                                                ConstRefVector, Data& data) {
  const Matrix& M = m_robot.mass(data);
  const Vector& h = m_robot.nonLinearEffects(data);

  m_constraint.matrix().setZero();
  for (unsigned int i = 0; i < m_activeAxes.size(); i++) {
    const unsigned int axis = m_activeAxes(i);
    m_constraint.matrix().row(i).leftCols(m_base_dim) =
        M.row(axis).leftCols(m_base_dim);

    if (m_use_projector) {
      m_projected_row.noalias() =
          M.row(axis).segment(m_base_dim, m_arm_dim) * m_projector;
      m_constraint.matrix().row(i).segment(m_base_dim, m_arm_dim) =
          m_projected_row;
    } else {
      m_constraint.matrix().row(i).segment(m_base_dim, m_arm_dim) =
          M.row(axis).segment(m_base_dim, m_arm_dim);
    }
    m_constraint.vector()(i) = m_reference(axis) - h(axis);
  }
  return m_constraint;
}

}  // namespace tasks
}  // namespace tsid
