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
  m_eq = m_u;
  m_in = 0;
  m_hqpData.resize(2);
  // m_hqpData[0].push_back(
  //     solvers::make_pair<double, std::shared_ptr<ConstraintBase> >(
  //         1.0, m_baseDynamics));
}

Data &InverseDynamicsFormulationAccForceOnShip::data() { return m_data; }

unsigned int InverseDynamicsFormulationAccForceOnShip::nVar() const
{
  return m_v;
}

unsigned int InverseDynamicsFormulationAccForceOnShip::nEq() const { return m_eq; }

unsigned int InverseDynamicsFormulationAccForceOnShip::nIn() const { return m_in; }

void InverseDynamicsFormulationAccForceOnShip::resizeHqpData()
{
  for (HQPData::iterator it = m_hqpData.begin(); it != m_hqpData.end(); it++)
  {
    for (ConstraintLevel::iterator itt = it->begin(); itt != it->end(); itt++)
    {
      itt->second->resize(itt->second->rows(), m_v);
    }
  }
}

template <class TaskLevelPointer>
void InverseDynamicsFormulationAccForceOnShip::addTask(TaskLevelPointer tl,
                                                       double weight,
                                                       unsigned int priorityLevel)
{
  if (priorityLevel > m_hqpData.size())
    m_hqpData.resize(priorityLevel);
  const ConstraintBase &c = tl->task.getConstraint();
  if (c.isEquality())
  {
    tl->constraint =
        std::make_shared<ConstraintEquality>(c.name(), c.rows(), m_v);
    if (priorityLevel == 0)
      m_eq += c.rows();
  }
  else // if(c.isInequality())
  {
    tl->constraint =
        std::make_shared<ConstraintInequality>(c.name(), c.rows(), m_v);
    if (priorityLevel == 0)
      m_in += c.rows();
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

  if (priorityLevel > m_hqpData.size())
    m_hqpData.resize(priorityLevel);

  const ConstraintBase &c = tl->task.getConstraint();
  if (c.isEquality())
  {
    tl->constraint =
        std::make_shared<ConstraintEquality>(c.name(), c.rows(), m_v);
    if (priorityLevel == 0)
      m_eq += c.rows();
  }
  else // an actuator bound becomes an inequality because actuator forces are
       // not in the problem variables
  {
    tl->constraint =
        std::make_shared<ConstraintInequality>(c.name(), c.rows(), m_v);
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
  return m_hqpData;
}


const HQPData &InverseDynamicsFormulationAccForceOnShip::computeProblemData(
    double time, ConstRefVector q, ConstRefVector v, ConstRefVector base_a)
{
  m_t = time;

  m_robot.computeAllTerms(m_data, q, v);

  const Matrix &M_a = m_robot.mass(m_data).bottomRows(m_v - m_u);
  const Matrix &M_vq = m_robot.mass(m_data).bottomRows(m_v - m_u).leftCols(m_u);
  Vector h_b = M_vq * base_a;
  const Vector &h_a =
      m_robot.nonLinearEffects(m_data).tail(m_v - m_u) + h_b;

  //  std::vector<TaskLevel*>::iterator it;
  //  for(it=m_taskMotions.begin(); it!=m_taskMotions.end(); it++)
  for (auto &it : m_taskMotions)
  {
    const ConstraintBase &c = it->task.compute(time, q, v, m_data);
    if (c.isEquality())
    {
      it->constraint->matrix().leftCols(m_v) = c.matrix();
      it->constraint->vector() = c.vector();
    }
    else if (c.isInequality())
    {
      it->constraint->matrix().leftCols(m_v) = c.matrix();
      it->constraint->lowerBound() = c.lowerBound();
      it->constraint->upperBound() = c.upperBound();
    }
    else
    {
      it->constraint->matrix().leftCols(m_v) = Matrix::Identity(m_v, m_v);
      it->constraint->lowerBound() = c.lowerBound();
      it->constraint->upperBound() = c.upperBound();
    }
  }

  for (auto &it : m_taskActuations)
  {
    const ConstraintBase &c = it->task.compute(time, q, v, m_data);
    if (c.isEquality())
    {
      it->constraint->matrix().leftCols(m_v).noalias() = c.matrix() * M_a;

      it->constraint->vector() = c.vector();
      it->constraint->vector().noalias() -= c.matrix() * h_a;
    }
    else if (c.isInequality())
    {
      it->constraint->matrix().leftCols(m_v).noalias() = c.matrix() * M_a;

      it->constraint->lowerBound() = c.lowerBound();
      it->constraint->lowerBound().noalias() -= c.matrix() * h_a;
      it->constraint->upperBound() = c.upperBound();
      it->constraint->upperBound().noalias() -= c.matrix() * h_a;
    }
    else
    {
      // NB: An actuator bound becomes an inequality
      it->constraint->matrix().leftCols(m_v) = M_a;
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

  const Matrix &M_a = m_robot.mass(m_data).bottomRows(m_v - m_u);
  const Vector &h_a =
      m_robot.nonLinearEffects(m_data).tail(m_v - m_u);
  m_dv = sol.x.head(m_v);
  m_tau = h_a;
  m_tau.noalias() += M_a * m_dv;
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
