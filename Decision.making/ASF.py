# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 12:18:35 2019

@author: babshava
"""

"""
 -Defining scalarization function for converting MOP into SOP
 -Using Wiezbicki1980 reference point scalarization function
    [WIERZBICKI1980] A. P. Wierzbicki, The use of reference objectives in multiobjective optimization, in: G. Fandel, T. Gal (Eds.),
    Multiple Criteria Decision Making, Theory and Applications, Vol. 177 of Lecture Notes in Economics and Mathematical Systems,
    Springer, 1980, pp. 468-486.
    
 
"""
import math
import numpy as np


def ASF(*args, **kwargs)
    x = args.T
    











#-----------------------------------------------

@mfunction("asfVal")
def ASF(x=None, StrucData=None):
    x = x.cT
    SurrogateData = StrucData.Surrogate
    ObjIndices = StrucData.ObjIndices
    NumObj = length(ObjIndices)
    y = zeros(1, NumObj)
    DecomposedBounds = StrucData.DecomposedBounds
    numPop = size(x, 1)
    range = DecomposedBounds(2, mslice[:]) - DecomposedBounds(1, mslice[:])
    x = ((x + 1) *elmul* repmat(range, numPop, 1)) / 2 + repmat(DecomposedBounds(1, mslice[:]), numPop, 1)
    zref = StrucData.z
    for Objective in mslice[1:NumObj]:
        y(Objective).lvalue = SurrogatePrediction(x, SurrogateData.SurrogateDataInfo(ObjIndices(Objective)))
        end

        asfVal = max((y - zref)) + (10 ** (-6)) * sum(y - zref, 2)
        end
        
# --------------------------------------
 # DESDEO scalarization function       
        class SimpleAchievementProblem(AchievementProblemBase):
    r"""
    Solves a simple form of achievement scalarizing function
    .. math::
       & \mbox{minimize}\ \
           & \displaystyle{
               \max_{i=1, \dots , k}
               \left\{\, \mu_i(f_i(\mathbf x) - q_i)\ \right\}}
           + \rho \sum_{i=1}^k \mu_i (f_i(\mathbf x)) \\
       & \mbox{subject to}\
           & {\bf{x}} \in S
    If ach_pen=True is passed to the constructor, the full achivement function
    is used as the penatly, causing us to instead solve[WIERZBICKI1980]_
    .. math::
       & \mbox{minimize}\ \
           & \displaystyle{
               \max_{i=1, \dots , k}
               \left\{\, \mu_i(f_i(\mathbf x) - q_i)\ \right\}}
           + \rho \sum_{i=1}^k \mu_i (f_i(\mathbf x)- q_i) \\
       & \mbox{subject to}\
           & {\bf{x}} \in S
    This is an abstract base class. Implementors should override `_get_rel` and
    `_set_scaling_weights`.
    References
    ----------
    [WIERZBICKI1980] A. P. Wierzbicki, The use of reference objectives in multiobjective optimization, in: G. Fandel, T. Gal (Eds.),
    Multiple Criteria Decision Making, Theory and Applications, Vol. 177 of Lecture Notes in Economics and Mathematical Systems,
    Springer, 1980, pp. 468-486.
    """

    def __init__(self, mo_problem: MOProblem, **kwargs) -> None:
        self.scaling_weights = None
        super().__init__(mo_problem, **kwargs)
        self.weights = [1.0] * self.problem.nobj
        if kwargs.get("ach_pen"):
            self.v_pen = v_ach
        else:
            self.v_pen = v_pen

    def _ach(self, objectives: List[List[float]]) -> List[float]:
        assert self.scaling_weights is not None
        return v_ach(objectives, np.array(self.scaling_weights), self._get_rel())

    def _augmentation(self, objectives: List[List[float]]) -> List[float]:
        assert self.scaling_weights is not None
        return self.v_pen(objectives, np.array(self.scaling_weights), self._get_rel())

    @abc.abstractmethod
    def _get_rel(self):
        pass
