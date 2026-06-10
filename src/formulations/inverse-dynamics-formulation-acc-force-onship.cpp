//
// Copyright (c) 2017 CNRS, NYU, MPI Tübingen
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

#include "tsid/formulations/inverse-dynamics-formulation-acc-force-onship.hpp"

#include "tsid/math/constraint-bound.hpp"
#include "tsid/math/constraint-inequality.hpp"

using namespace tsid;
using namespace math;
using namespace tasks;
using namespace contacts;
using namespace solvers;
using namespace std;

typedef pinocchio::Data Data;

InverseDynamicsFormulationAccForceOnShip::InverseDynamicsFormulationAccForceOnShip(
    const std::string &name, RobotWrapper &robot, bool verbose)
    : InverseDynamicsFormulationBase(name, robot, verbose),
      m_data(robot.model()),
      m_solutionDecoded(false)
{
  m_t = 0.0;
  m_v = robot.nv();
  m_u = robot.nv() - robot.na();
  m_eq = 0;
  m_in = 0;
  m_hqpData.resize(2);
  m_base_acc.setZero(m_u);
  m_dv.setZero(m_v);
  m_f.setZero(0);
  m_tau.setZero(m_v - m_u);
}

Data &InverseDynamicsFormulationAccForceOnShip::data() { return m_data; }

unsigned int InverseDynamicsFormulationAccForceOnShip::nVar() const
{
  return m_v - m_u;
}

unsigned int InverseDynamicsFormulationAccForceOnShip::nEq() const { return m_eq; }

unsigned int InverseDynamicsFormulationAccForceOnShip::nIn() const { return m_in; }

void InverseDynamicsFormulationAccForceOnShip::resizeHqpData()
{
  for (HQPData::iterator it = m_hqpData.begin(); it != m_hqpData.end(); it++)
  {
    for (ConstraintLevel::iterator itt = it->begin(); itt != it->end(); itt++)
    {
      itt->second->resize(itt->second->rows(), nVar());
    }
  }
}

template <class TaskLevelPointer>
void InverseDynamicsFormulationAccForceOnShip::addTask(TaskLevelPointer tl,
                                                       double weight,
                                                       unsigned int priorityLevel)
{
  if (priorityLevel >= m_hqpData.size())
    m_hqpData.resize(priorityLevel + 1);
  const ConstraintBase &c = tl->task.getConstraint();
  const unsigned int rows =
      (c.isBound() && c.rows() == m_v) ? nVar() : c.rows();
  if (c.isEquality())
  {
    tl->constraint =
        std::make_shared<ConstraintEquality>(c.name(), rows, nVar());
    if (priorityLevel == 0)
      m_eq += rows;
  }
  else // if(c.isInequality())
  {
    tl->constraint =
        std::make_shared<ConstraintInequality>(c.name(), rows, nVar());
    if (priorityLevel == 0)
      m_in += rows;
  }
  // don't use bounds for now because EiQuadProg doesn't exploit them anyway
  //  else
  //    tl->constraint = new ConstraintBound(c.name(), m_v+m_k);
  m_hqpData[priorityLevel].push_back(
      make_pair<double, std::shared_ptr<ConstraintBase>>(weight,
                                                         tl->constraint));
}

bool InverseDynamicsFormulationAccForceOnShip::addMotionTask(
    TaskMotion &task, double weight, unsigned int priorityLevel,
    double transition_duration)
{
  PINOCCHIO_CHECK_INPUT_ARGUMENT(
      weight >= 0.0, "The weight needs to be positive or equal to 0");
  PINOCCHIO_CHECK_INPUT_ARGUMENT(
      transition_duration >= 0.0,
      "The transition duration needs to be greater than or equal to 0");

  auto tl = std::make_shared<TaskLevel>(task, priorityLevel);
  m_taskMotions.push_back(tl);
  addTask(tl, weight, priorityLevel);

  return true;
}

bool InverseDynamicsFormulationAccForceOnShip::addActuationTask(
    TaskActuation &task, double weight, unsigned int priorityLevel,
    double transition_duration)
{
  PINOCCHIO_CHECK_INPUT_ARGUMENT(
      weight >= 0.0, "The weight needs to be positive or equal to 0");
  PINOCCHIO_CHECK_INPUT_ARGUMENT(
      transition_duration >= 0.0,
      "The transition duration needs to be greater than or equal to 0");

  auto tl = std::make_shared<TaskLevel>(task, priorityLevel);
  m_taskActuations.push_back(tl);

  if (priorityLevel >= m_hqpData.size())
    m_hqpData.resize(priorityLevel + 1);

  const ConstraintBase &c = tl->task.getConstraint();
  if (c.isEquality())
  {
    tl->constraint =
        std::make_shared<ConstraintEquality>(c.name(), c.rows(), nVar());
    if (priorityLevel == 0)
      m_eq += c.rows();
  }
  else // an actuator bound becomes an inequality because actuator forces are
       // not in the problem variables
  {
    tl->constraint =
        std::make_shared<ConstraintInequality>(c.name(), c.rows(), nVar());
    if (priorityLevel == 0)
      m_in += c.rows();
  }

  m_hqpData[priorityLevel].push_back(
      make_pair<double, std::shared_ptr<ConstraintBase>>(weight,
                                                         tl->constraint));

  return true;
}

bool InverseDynamicsFormulationAccForceOnShip::updateTaskWeight(
    const std::string &task_name, double weight)
{
  ConstraintLevel::iterator it;
  // do not look into first priority level because weights do not matter there
  for (unsigned int i = 1; i < m_hqpData.size(); i++)
  {
    for (it = m_hqpData[i].begin(); it != m_hqpData[i].end(); it++)
    {
      if (it->second->name() == task_name)
      {
        it->first = weight;
        return true;
      }
    }
  }
  return false;
}

const HQPData &InverseDynamicsFormulationAccForceOnShip::computeProblemData(
    double time, ConstRefVector q, ConstRefVector v)
{
  m_base_acc.setZero();
  return computeProblemData(time, q, v, m_base_acc);
}


const HQPData &InverseDynamicsFormulationAccForceOnShip::computeProblemData(
    double time, ConstRefVector q, ConstRefVector v, ConstRefVector base_a)
{
  m_t = time;
  PINOCCHIO_CHECK_INPUT_ARGUMENT(
      base_a.size() == m_u,
      "The size of the base acceleration vector needs to equal " +
          std::to_string(m_u));
  m_base_acc = base_a;

  m_robot.computeAllTerms(m_data, q, v);

  const unsigned int na = nVar();
  const Matrix &M_actuated = m_robot.mass(m_data).bottomRows(na);
  const auto M_ab = M_actuated.leftCols(m_u);
  const auto M_aa = M_actuated.rightCols(na);
  const Vector h_a =
      m_robot.nonLinearEffects(m_data).tail(na) + M_ab * m_base_acc;

  //  std::vector<TaskLevel*>::iterator it;
  //  for(it=m_taskMotions.begin(); it!=m_taskMotions.end(); it++)
  for (auto &it : m_taskMotions)
  {
    const ConstraintBase &c = it->task.compute(time, q, v, m_data);
    if (c.isEquality())
    {
      const Vector base_term = c.matrix().leftCols(m_u) * m_base_acc;
      it->constraint->matrix() = c.matrix().rightCols(na);
      it->constraint->vector() = c.vector() - base_term;
    }
    else if (c.isInequality())
    {
      const Vector base_term = c.matrix().leftCols(m_u) * m_base_acc;
      it->constraint->matrix() = c.matrix().rightCols(na);
      it->constraint->lowerBound() = c.lowerBound() - base_term;
      it->constraint->upperBound() = c.upperBound() - base_term;
    }
    else
    {
      if (c.rows() == m_v && it->constraint->rows() == na)
      {
        it->constraint->matrix() = Matrix::Identity(na, na);
        it->constraint->lowerBound() = c.lowerBound().tail(na);
        it->constraint->upperBound() = c.upperBound().tail(na);
      }
      else
      {
        const Vector base_term = c.matrix().leftCols(m_u) * m_base_acc;
        it->constraint->matrix() = c.matrix().rightCols(na);
        it->constraint->lowerBound() = c.lowerBound() - base_term;
        it->constraint->upperBound() = c.upperBound() - base_term;
      }
    }
  }

  for (auto &it : m_taskActuations)
  {
    const ConstraintBase &c = it->task.compute(time, q, v, m_data);
    if (c.isEquality())
    {
      it->constraint->matrix().noalias() = c.matrix() * M_aa;

      it->constraint->vector() = c.vector();
      it->constraint->vector().noalias() -= c.matrix() * h_a;
    }
    else if (c.isInequality())
    {
      it->constraint->matrix().noalias() = c.matrix() * M_aa;

      it->constraint->lowerBound() = c.lowerBound();
      it->constraint->lowerBound().noalias() -= c.matrix() * h_a;
      it->constraint->upperBound() = c.upperBound();
      it->constraint->upperBound().noalias() -= c.matrix() * h_a;
    }
    else
    {
      // NB: An actuator bound becomes an inequality
      it->constraint->matrix() = M_aa;
      it->constraint->lowerBound() = c.lowerBound() - h_a;
      it->constraint->upperBound() = c.upperBound() - h_a;
    }
  }

  m_solutionDecoded = false;

  return m_hqpData;
}

bool InverseDynamicsFormulationAccForceOnShip::decodeSolution(const HQPOutput &sol)
{
  if (m_solutionDecoded)
    return true;

  const unsigned int na = nVar();
  const Matrix &M_actuated = m_robot.mass(m_data).bottomRows(na);
  const auto M_ab = M_actuated.leftCols(m_u);
  const auto M_aa = M_actuated.rightCols(na);
  const Vector h_a =
      m_robot.nonLinearEffects(m_data).tail(na) + M_ab * m_base_acc;
  m_dv.head(m_u) = m_base_acc;
  m_dv.tail(na) = sol.x.head(na);
  m_tau = h_a;
  m_tau.noalias() += M_aa * m_dv.tail(na);
  m_solutionDecoded = true;
  return true;
}

const Vector &InverseDynamicsFormulationAccForceOnShip::getActuatorForces(
    const HQPOutput &sol)
{
  decodeSolution(sol);
  return m_tau;
}

const Vector &InverseDynamicsFormulationAccForceOnShip::getAccelerations(
    const HQPOutput &sol)
{
  decodeSolution(sol);
  return m_dv;
}

bool InverseDynamicsFormulationAccForceOnShip::removeTask(const std::string &taskName,
                                                          double)
{
#ifndef NDEBUG
  bool taskFound = removeFromHqpData(taskName);
  assert(taskFound);
#else
  removeFromHqpData(taskName);
#endif

  for (auto it = m_taskMotions.begin(); it != m_taskMotions.end(); it++)
  {
    if ((*it)->task.name() == taskName)
    {
      if ((*it)->priority == 0)
      {
        if ((*it)->constraint->isEquality())
          m_eq -= (*it)->constraint->rows();
        else if ((*it)->constraint->isInequality())
          m_in -= (*it)->constraint->rows();
      }
      m_taskMotions.erase(it);
      return true;
    }
  }

  for (auto it = m_taskActuations.begin(); it != m_taskActuations.end(); it++)
  {
    if ((*it)->task.name() == taskName)
    {
      if ((*it)->priority == 0)
      {
        if ((*it)->constraint->isEquality())
          m_eq -= (*it)->constraint->rows();
        else
          m_in -= (*it)->constraint->rows();
      }
      m_taskActuations.erase(it);
      return true;
    }
  }
  return false;
}

bool InverseDynamicsFormulationAccForceOnShip::removeFromHqpData(
    const std::string &name)
{
  bool found = false;
  for (HQPData::iterator it = m_hqpData.begin();
       !found && it != m_hqpData.end(); it++)
  {
    for (ConstraintLevel::iterator itt = it->begin();
         !found && itt != it->end(); itt++)
    {
      if (itt->second->name() == name)
      {
        it->erase(itt);
        return true;
      }
    }
  }
  return false;
}

bool InverseDynamicsFormulationAccForceOnShip::getContactForces(
    const std::string &name, const HQPOutput &sol, RefVector f)
{
  return false;
}

bool InverseDynamicsFormulationAccForceOnShip::addMeasuredForce(MeasuredForceBase &measuredForce) { return false; }
bool InverseDynamicsFormulationAccForceOnShip::removeRigidContact(const std::string &contactName,
                                                                  double transition_duration) { return false; }

bool InverseDynamicsFormulationAccForceOnShip::removeMeasuredForce(const std::string &measuredForceName) { return false; }
TSID_DEPRECATED bool InverseDynamicsFormulationAccForceOnShip::addRigidContact(ContactBase &contact) { return false; }

bool InverseDynamicsFormulationAccForceOnShip::updateRigidContactWeights(const std::string &contact_name,
                                                                         double force_regularization_weight,
                                                                         double motion_weight) { return false; }
bool InverseDynamicsFormulationAccForceOnShip::addRigidContact(ContactBase &contact, double force_regularization_weight,
                                                               double motion_weight,
                                                               unsigned int motion_priority_level) { return false; }
const Vector &InverseDynamicsFormulationAccForceOnShip::getContactForces(
    const HQPOutput &sol)
{
  decodeSolution(sol);
  return m_f;
}

bool InverseDynamicsFormulationAccForceOnShip::addForceTask(
    TaskContactForce &task, double weight, unsigned int priorityLevel,
    double transition_duration)
{
  return true;
}
