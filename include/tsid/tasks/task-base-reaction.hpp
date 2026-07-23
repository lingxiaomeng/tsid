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

#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/spatial/motion.hpp>
#include <pinocchio/spatial/se3.hpp>

#include <vector>

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
  TaskBaseReaction(const std::string& name, RobotWrapper& robot,
                   const pinocchio::Model& free_flyer_model);

  int dim() const override;

  const ConstraintBase& compute(const double t, ConstRefVector q,
                                ConstRefVector v, Data& data) override;

  const ConstraintBase& getConstraint() const override;

  void setReference(math::ConstRefVector reference);
  void setMask(math::ConstRefVector mask) override;
  void setProjector(math::ConstRefMatrix projector);
  void disableProjector();

  void setBaseKinematics(const pinocchio::SE3& placement_world_base,
                         const pinocchio::Motion& velocity_body,
                         const pinocchio::Motion& classical_acceleration_body);
  void setBaseAcceleration(math::ConstRefVector spatial_acceleration_body);
  void setMountToComTransform(math::ConstRefMatrix rotation_com_mount,
                              math::ConstRefVector position_com_to_mount_com);
  void setRollPitchTorqueScale(double roll_scale, double pitch_scale);
  void setObjectiveWeights(double reaction_weight, double gravity_weight);
  void setGravityPredictionHorizon(double prediction_horizon);

  void updateReaction(math::ConstRefVector arm_acceleration);
  double validateAffineModel(math::ConstRefVector arm_acceleration);

  const Vector& reactionAtMount() const;
  const Vector& reactionAtCom() const;
  const Eigen::Vector2d& rollPitchTorque() const;
  const Matrix& reactionMatrixAtMount() const;
  const Vector& reactionOffsetAtMount() const;
  const Matrix& reactionMatrixAtCom() const;
  const Vector& reactionOffsetAtCom() const;
  const Eigen::Matrix<double, 2, Eigen::Dynamic>& rollPitchMatrix() const;
  const Eigen::Vector2d& rollPitchOffset() const;
  const Eigen::Vector2d& gravityRollPitchTorque() const;
  const Eigen::Vector2d& predictedGravityRollPitchTorque() const;
  double computationTimeUs() const;

 protected:
  struct JointIndexMap {
    pinocchio::JointIndex input_joint;
    pinocchio::JointIndex free_flyer_joint;
    int input_idx_q;
    int free_flyer_idx_q;
    int nq;
    int input_idx_v;
    int free_flyer_idx_v;
    int nv;
    int actuated_idx_v;
  };

  void buildJointIndexMap();
  void updateFreeFlyerState(ConstRefVector q, ConstRefVector v);

  unsigned int m_base_dim;
  unsigned int m_arm_dim;
  pinocchio::Model m_model_ff;
  pinocchio::Data m_data_ff;
  pinocchio::JointIndex m_input_base_joint;
  pinocchio::JointIndex m_ff_base_joint;
  std::vector<JointIndexMap> m_joint_index_map;

  VectorXi m_activeAxes;
  Vector m_reference;
  Vector m_q_ff;
  Vector m_v_ff;
  Vector m_a_ff;
  Vector m_base_acceleration;
  Eigen::Matrix<double, 6, 6> m_M_BB;
  Matrix m_M_Bq;
  Vector m_h_B;
  Vector m_h_com;
  Vector m_tau_rnea;
  Vector m_affine_check;
  Matrix m_gravity_partial_dq;
  Matrix m_gravity_jacobian_mount;
  Matrix m_gravity_jacobian_com;
  Matrix m_projected_gravity_jacobian_com;
  Vector m_gravity_reaction_mount;
  Vector m_gravity_reaction_com;
  Vector m_arm_velocity;
  Eigen::Vector2d m_gravity_roll_pitch_torque;
  Eigen::Vector2d m_gravity_prediction_offset;
  Eigen::Vector2d m_predicted_gravity_roll_pitch_torque;

  Eigen::Matrix<double, 6, 6> m_wrench_mount_to_com;
  Eigen::Matrix<double, 6, 6> m_base_matrix_com;
  Eigen::Matrix<double, 6, 1> m_axis_scale;
  Matrix m_G_mount;
  Vector m_w0_mount;
  Matrix m_G_com;
  Vector m_w0_com;
  Eigen::Matrix<double, 2, Eigen::Dynamic> m_G_roll_pitch;
  Eigen::Vector2d m_b_roll_pitch;
  Vector m_reaction_mount;
  Vector m_reaction_com;
  Eigen::Vector2d m_roll_pitch_torque;
  double m_computation_time_us;

  bool m_use_projector;
  Matrix m_projector;
  Matrix m_projected_G_com;
  double m_reaction_weight_sqrt;
  double m_gravity_weight_sqrt;
  double m_gravity_prediction_horizon;
  ConstraintEquality m_constraint;
};

}  // namespace tasks
}  // namespace tsid

#endif  // ifndef __invdyn_task_base_reaction_hpp__
