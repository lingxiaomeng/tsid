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

#include <pinocchio/fwd.hpp>

#include <tsid/tasks/task-joint-bounds.hpp>
#include "tsid/robots/robot-wrapper.hpp"

namespace tsid {
namespace tasks {
using namespace math;
using namespace trajectories;
using namespace pinocchio;

TaskJointBounds::TaskJointBounds(const std::string& name, RobotWrapper& robot,
                                 double dt)
    : TaskMotion(name, robot),
      m_constraint(name, robot.nv()),
      m_dt(dt),
      m_nv(robot.nv()),
      m_na(robot.na()) {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(dt > 0.0, "dt needs to be positive");
  m_v_lb = -1e10 * Vector::Ones(m_na);
  m_v_ub = +1e10 * Vector::Ones(m_na);
  m_a_lb = -1e10 * Vector::Ones(m_na);
  m_a_ub = +1e10 * Vector::Ones(m_na);
  m_ddq_max_due_to_vel.setZero(m_na);
  m_ddq_min_due_to_vel.setZero(m_na);

  int offset = m_nv - m_na;
  for (int i = 0; i < offset; i++) {
    m_constraint.upperBound()(i) = 1e10;
    m_constraint.lowerBound()(i) = -1e10;
  }
}

int TaskJointBounds::dim() const { return m_nv; }

const Vector& TaskJointBounds::getPositionLowerBounds() const { return m_q_lb; }

const Vector& TaskJointBounds::getPositionUpperBounds() const { return m_q_ub; }

const Vector& TaskJointBounds::getAccelerationLowerBounds() const {
  return m_a_lb;
}

const Vector& TaskJointBounds::getAccelerationUpperBounds() const {
  return m_a_ub;
}

const Vector& TaskJointBounds::getVelocityLowerBounds() const { return m_v_lb; }

const Vector& TaskJointBounds::getVelocityUpperBounds() const { return m_v_ub; }

void TaskJointBounds::setTimeStep(double dt) {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(dt > 0.0, "dt needs to be positive");
  m_dt = dt;
}

void TaskJointBounds::setPositionBounds(ConstRefVector lower,
                                       ConstRefVector upper) {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(
      m_robot.nq_actuated() == m_na,
      "Position bounds require Euclidean actuated joint coordinates");
  PINOCCHIO_CHECK_INPUT_ARGUMENT(
      lower.size() == m_na && upper.size() == m_na,
      "The position bounds vectors need to have size " + std::to_string(m_na));
  PINOCCHIO_CHECK_INPUT_ARGUMENT(
      lower.allFinite() && upper.allFinite() &&
          (lower.array() <= upper.array()).all(),
      "Position bounds must be finite and ordered");
  m_q_lb = lower;
  m_q_ub = upper;
  m_impose_position_bounds = true;
}

void TaskJointBounds::setVelocityBounds(ConstRefVector lower,
                                        ConstRefVector upper) {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(
      lower.size() == m_na,
      "The size of the lower velocity bounds vector needs to equal " +
          std::to_string(m_na));
  PINOCCHIO_CHECK_INPUT_ARGUMENT(
      upper.size() == m_na,
      "The size of the upper velocity bounds vector needs to equal " +
          std::to_string(m_na));
  m_v_lb = lower;
  m_v_ub = upper;
}

void TaskJointBounds::setAccelerationBounds(ConstRefVector lower,
                                            ConstRefVector upper) {
  PINOCCHIO_CHECK_INPUT_ARGUMENT(
      lower.size() == m_na,
      "The size of the lower acceleration bounds vector needs to equal " +
          std::to_string(m_na));
  PINOCCHIO_CHECK_INPUT_ARGUMENT(
      upper.size() == m_na,
      "The size of the upper acceleration bounds vector needs to equal " +
          std::to_string(m_na));
  m_a_lb = lower;
  m_a_ub = upper;
}

const ConstraintBase& TaskJointBounds::getConstraint() const {
  return m_constraint;
}

void TaskJointBounds::setMask(ConstRefVector mask) { m_mask = mask; }

const ConstraintBase& TaskJointBounds::compute(const double, ConstRefVector q,
                                               ConstRefVector v, Data&) {
  // compute min/max joint acc imposed by velocity limits
  m_ddq_max_due_to_vel = (m_v_ub - v.tail(m_na)) / m_dt;
  m_ddq_min_due_to_vel = (m_v_lb - v.tail(m_na)) / m_dt;

  // Intersect hard velocity, acceleration, and optional position bounds.
  int offset = m_nv - m_na;
  for (int i = 0; i < m_na; i++) {
    // TODO: use mask here
    m_constraint.upperBound()(offset + i) =
        std::min(m_ddq_max_due_to_vel(i), m_a_ub(i));
    m_constraint.lowerBound()(offset + i) =
        std::max(m_ddq_min_due_to_vel(i), m_a_lb(i));
    if (m_impose_position_bounds) {
      const double q_free = q.tail(m_na)(i) + m_dt * v.tail(m_na)(i);
      const double position_scale = 2.0 / (m_dt * m_dt);
      m_constraint.upperBound()(offset + i) =
          std::min(m_constraint.upperBound()(offset + i),
                   position_scale * (m_q_ub(i) - q_free));
      m_constraint.lowerBound()(offset + i) =
          std::max(m_constraint.lowerBound()(offset + i),
                   position_scale * (m_q_lb(i) - q_free));
      // Preserve an empty intersection: relaxing it would discard a hard bound.
    }
  }
  return m_constraint;
}

}  // namespace tasks
}  // namespace tsid
