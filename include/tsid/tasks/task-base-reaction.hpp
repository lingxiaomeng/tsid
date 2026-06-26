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

#ifndef __invdyn_task_base_reaction_hpp__
#define __invdyn_task_base_reaction_hpp__

#include <tsid/tasks/task-motion.hpp>
#include <tsid/math/constraint-equality.hpp>

namespace tsid {
namespace tasks {

class TaskBaseReaction : public TaskMotion {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef math::Vector Vector;
  typedef math::Matrix Matrix;
  typedef math::VectorXi VectorXi;
  typedef math::ConstraintEquality ConstraintEquality;
  typedef pinocchio::Data Data;

  TaskBaseReaction(const std::string& name, RobotWrapper& robot);

  int dim() const override;

  const ConstraintBase& compute(const double t, ConstRefVector q,
                                ConstRefVector v, Data& data) override;

  const ConstraintBase& getConstraint() const override;

  void setReference(math::ConstRefVector reference);
  void setMask(math::ConstRefVector mask) override;
  void setProjector(math::ConstRefMatrix projector);
  void disableProjector();

 protected:
  unsigned int m_base_dim;
  unsigned int m_arm_dim;
  VectorXi m_activeAxes;
  Vector m_reference;
  bool m_use_projector;
  Matrix m_projector;
  Matrix m_projected_row;
  ConstraintEquality m_constraint;
};

}  // namespace tasks
}  // namespace tsid

#endif  // ifndef __invdyn_task_base_reaction_hpp__
