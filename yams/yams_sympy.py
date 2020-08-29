"""
Reference:
     [1]: Branlard, Flexible multibody dynamics using joint coordinates and the Rayleigh-Ritz approximation: the general framework behind and beyond Flex, Wind Energy, 2019
"""
import numpy as np
import sympy
from sympy import Symbol, symbols
from sympy import Matrix, Function, diff
from sympy.printing import lambdarepr
from sympy import init_printing
from sympy import lambdify
#from sympy.abc import *
from sympy import trigsimp
from sympy import cos,sin
from sympy import zeros

from sympy.physics.mechanics import Body as SympyBody
from sympy.physics.mechanics import RigidBody as SympyRigidBody
from sympy.physics.mechanics import Point, ReferenceFrame, inertia, dynamicsymbols
from sympy.physics.mechanics.functions import msubs

from sympy.physics.vector import init_vprinting, vlatex
init_vprinting(use_latex='mathjax', pretty_print=False)

display=lambda x: sympy.pprint(x, use_unicode=False,wrap_line=False)

__all__ = ['YAMSBody','YAMSInertialBody','YAMSRigidBody','YAMSFlexibleBody']
__all__+= ['Body','RigidBody','GroundBody'] # Old implementation
__all__+= ['skew', 'rotToDCM', 'DCMtoOmega']
__all__+= ['kane_frstar','kane_fr','kane_fr_alt','kane_frstar_alt']

# --------------------------------------------------------------------------------}
# --- Helper functions 
# --------------------------------------------------------------------------------{
def colvec(v): 
    return Matrix([[v[0]],[v[1]],[v[2]]])
def cross(V1,V2):
    return [V1[1]*V2[2]-V1[2]*V2[1], V1[2]*V2[0]-V1[0]*V2[2], (V1[0]*V2[1]-V1[1]*V2[0]) ]
def eye(n): 
    return Matrix( np.eye(n).astype(int) )

def ensureMat(x, nr, nc):
    """ Ensures that the input is a matrix of shape nr, nc"""
    if not isinstance(x,Matrix):
        x=Matrix(x)
    return x.reshape(nr, nc)

def ensureList(x, nr):
    """ Ensures that the input is a list of length nr"""
    x = list(x)
    if len(x)!=nr:
        raise Exception('Wrong dimension, got {}, expected {}'.format(len(x),nr))
    return x
            
def coord2vec(M31, e):
    """ Ugly conversion from a matrix or vector coordinates (implicit frame) to a vector (in a given frame) """
    M31 = ensureList(M31, 3)
    return M31[0] * e.x + M31[1] * e.y + M31[2] * e.z

            
def skew(x):
    """ Returns the skew symmetric matrix M, such that: cross(x,v) = M v """
    #S = Matrix(np.zeros((3,3)).astype(int))
    if hasattr(x,'shape') and len(x.shape)==2:
        if x.shape[0]==3:
            return Matrix(np.array([[0, -x[2,0], x[1,0]],[x[2,0],0,-x[0,0]],[-x[1,0],x[0,0],0]]))
        else:
            raise Exception('fSkew expect a vector of size 3 or matrix of size 3x1, got {}'.format(x.shape))
    else:
        return Matrix(np.array([[0, -x[2], x[1]],[x[2],0,-x[0]],[-x[1],x[0],0]]))

def rotToDCM(rot_type, rot_amounts, rot_order=None):
    """
    return matrix from a ref_frame to another frame rotated by specified amounts
    
    New type added: SmallRot
    
    see sympy.orientnew
        rotToDCM(frame, 'Axis', (3, N.x)       )
        rotToDCM(frame, 'Body', (x,y,z), 'XYZ' )
        rotToDCM(frame, 'DCM' , M )
        rotToDCM(frame, 'SmallRot' , (x,y,z) )
    """
    ref_frame = ReferenceFrame('dummyref')
    if rot_type =='SmallRot':
            M=-skew(rot_amounts)
            M[0,0]=1
            M[1,1]=1
            M[2,2]=1
            return M
    elif rot_type in ['Body','Space']:
        frame = ref_frame.orientnew('dummy', rot_type, rot_amounts, rot_order)
    else:
        frame = ref_frame.orientnew('dummy', rot_type, rot_amounts)
    return frame.dcm(ref_frame) # from parent to frame
    
def DCMtoOmega(DCM, ref_frame=None):
    """
    Given a DCM matrix, returns the rotational velocity: omega = R' * R^t

        DCM = ref_frame.dcm(body.frame) # from body to inertial
           -> Omega of body wrt ref_frame expressed in ref frame

    If ref_frame is None, only the coordinates of omega are given 
    otherwise, a vector is returned, expressed in ref_frame

    """
    t = dynamicsymbols._t
    OmSkew = (DCM.diff(t) *  DCM.transpose()).simplify()
    if ref_frame is None:
        return (OmSkew[2,1], OmSkew[0,2], OmSkew[1,0])
    else:
        return OmSkew[2,1] * ref_frame.x + OmSkew[0,2]*ref_frame.y + OmSkew[1,0] * ref_frame.z


# --------------------------------------------------------------------------------}
# ---  
# --------------------------------------------------------------------------------{
class Taylor(object):
    """ 
    A Taylor object contains a Taylor expansion of a variable as function of q
        M = M^0 + \sum_j=1^nq M^1_j q_j
    where M, M^0, M^1_j are matrices of dimension nr x nc
    See Wallrapp 1993/1994
    """
    def __init__(self, bodyname, varname, nr, nc, nq, rname=None, cname=None, q=None, order=2):
        if rname is None:
            rname=list(np.arange(nr)+1)
        if cname is None:
            cname=list(np.arange(nr)+1)
        if len(cname)!=nc:
            raise Exception('cname length should match nc for Taylor {} {}'.format(bodyname, varname))
        if len(rname)!=nr:
            raise Exception('rname length should match nr for Taylor {} {}'.format(bodyname, varname))
            
        self.varname=varname
        self.bodyname=bodyname
            
        self.M0=Matrix(np.zeros((nr,nc)).astype(int))
        for i in np.arange(nr):
            for j in np.arange(nc):
                self.M0[i,j] = symbols('{}^0_{}_{}{}'.format(varname,bodyname,rname[i],cname[j])) 
                
        if order==2: 
            self.M1=[]
            for k in np.arange(nq):
                self.M1.append(Matrix(np.zeros((nr,nc)).astype(int)))
                for i in np.arange(nr):
                    for j in np.arange(nc):
                        self.M1[k][i,j] = symbols('{}^1_{}_{}_{}{}'.format(varname,k+1,bodyname,rname[i],cname[j])) 
    def eval(self, q):
        M = self.M0
        if hasattr(self,'M1'): 
            nq = len(self.M1)
            if len(q)!=nq:
                raise Exception('Inconsistent dimension between q ({}) and M1 ({}) for Taylor {} {}'.format(len(q),nq,self.bodyname, self.varname))
            for k in np.arange(nq):
                M +=self.M1[k]*q[k]
        return M
    
    def setOrder(self,order):
        if order==1:
            if hasattr(self,'M1'): 
                del self.M1
        else:
            raise Exception('set order for now mainly removes the 2nd order term')

#Me = Taylor('T','Me', 3, 3, nq=2, rname='xyz', cname='xyz')
#Me.M1
#Me.eval([x,y])
#Md = Taylor('T','M_d', 3, 1, nq=2, rname='xyz', cname=[''])
#skew(Md.M0)
# --------------------------------------------------------------------------------}
# --- Connections 
# --------------------------------------------------------------------------------{
class Connection():
    def __init__(self,Type,RelPoint=None,RelOrientation=None,JointRotations=None):
        if RelOrientation is None:
            RelOrientation=eye(3)
        if RelPoint is None:
            RelPoint=colvec([0,0,0])

        self.Type=Type
        
        self.s_C_0_inB = RelPoint
        self.s_C_inB   = self.s_C_0_inB
        self.R_ci_0    = RelOrientation
        self.R_ci      = self.R_ci_0     

        if self.Type=='Rigid':
            self.nj=0
        elif self.Type=='SphericalJoint':
            self.JointRotations=JointRotations;
            self.nj=len(self.JointRotations);
        else:
            raise NotImplementedError()

    def updateKinematics(j,q):
        j.B_ci=Matrix(np.zeros((6,j.nj)))
        if j.Type=='Rigid':
            j.R_ci=j.R_ci_0
        elif j.Type=='SphericalJoint':
            R=eye(3)
            myq    = q   [j.I_DOF,0];
            #myqdot = qdot[j.I_DOF];

            for ir,rot in enumerate(j.JointRotations):
                if rot=='x':
                    I=np.array([1,0,0])
                    Rj=R_x( myq[ir] )
                elif rot=='y':
                    I=np.array([0,1,0])
                    Rj=R_y( myq[ir] )
                elif rot=='z':
                    I=np.array([0,0,1])
                    Rj=R_z( myq[ir] )
                else:
                    raise Exception()
                # Setting Bhat column by column
                j.B_ci[3:,ir] = np.dot(R,I) # NOTE: needs to be done before R updates
                # Updating rotation matrix
                R      = np.dot(R , Rj )
                j.R_ci = Matrix(np.dot(R, j.R_ci_0 ))

def exprHasFunction(expr):
    """ return True if a sympy expression contains a function"""
    if hasattr(expr, 'atoms'): 
        return len(expr.atoms(Function))>0
    else:
        return False



# --------------------------------------------------------------------------------}
# --- Bodies 
# --------------------------------------------------------------------------------{
class YAMSBody(object):
    def __init__(self, name):
        """
           Origin point have no velocities in the body frame! 
        """
        self.frame     = ReferenceFrame('e_'+name)
        self.origin    = Point('O_'+name)
        self.masscenter= Point('G_'+name)
        self.name=name
        self.origin.set_vel(self.frame,0*self.frame.x)
        
        self.parent = None # Parent body, assuming a tree structure
        self.children = [] # children bodies
        self.inertial_frame = None # storing the typical inertial frame use for computation
        self.inertial_origin = None # storing the typical inertial frame use for computation

    def __repr__(self):
        s='<{} object "{}" with attributes:>\n'.format(type(self).__name__,self.name)
        s+=' - origin:       {}\n'.format(self.origin)
        s+=' - frame:        {}\n'.format(self.frame)
        return s
        
    def ang_vel_in(self,frame_or_body):
        """ Angular velocity of body wrt to another frame or body
        This is just a wrapper for the ReferenceFrame ang_vel_in function
        """
        if isinstance(frame_or_body,ReferenceFrame):
            return self.frame.ang_vel_in(frame_or_body)
        else:
            if issubclass(type(frame_or_body),YAMSBody):
                return self.frame.ang_vel_in(frame_or_body.frame)
            else:
                raise Exception('Unknown class type, use ReferenceFrame of YAMSBody as argument')
                
    def connectTo(parent,child,type='Rigid', rel_pos=None, rot_type='Body', rot_amounts=None, rot_order=None, dynamicAllowed=False):
        # register parent/child relationship
        child.parent = parent
        parent.children.append(child)
        if isinstance(parent, YAMSInertialBody):
            parent.inertial_frame = parent.frame
            child.inertial_frame = parent.frame
            parent.inertial_origin = parent.origin
            child.inertial_origin = parent.origin
        else:
            if parent.inertial_frame is None:
                raise Exception('Parent body was not connected to an inertial frame. Bodies needs to be connected in order, starting from inertial frame.')
            else:
                child.inertial_frame  = parent.inertial_frame # the same frame is used for all connected bodies
                child.inertial_origin = parent.origin

        if rel_pos is None or len(rel_pos)!=3:
            raise Exception('rel_pos needs to be an array of size 3')

        pos = 0 * parent.frame.x
        vel = 0 * parent.frame.x

        t = dynamicsymbols._t

        if type=='Free':
            # --- "Free", "floating" connection
            if not isinstance(parent, YAMSInertialBody):
                raise Exception('Parent needs to be inertial body for a free connection')
            # Defining relative position and velocity of child wrt parent
            for d,e in zip(rel_pos[0:3], (parent.frame.x, parent.frame.y, parent.frame.z)):
                if d is not None:
                    pos += d * e
                    vel += diff(d,t) * e

        elif type=='Rigid':
            # Defining relative position and velocity of child wrt parent
            for d,e in zip(rel_pos[0:3], (parent.frame.x, parent.frame.y, parent.frame.z)):
                if d is not None:
                    pos += d * e
                    if exprHasFunction(d) and not dynamicAllowed:
                        raise Exception('Position variable cannot be a dynamic variable for a rigid connection: variable {}'.format(d))
                    if dynamicAllowed:
                        vel += diff(d,t) * e
        elif type=='Joint':
            # Defining relative position and velocity of child wrt parent
            for d,e in zip(rel_pos[0:3], (parent.frame.x, parent.frame.y, parent.frame.z)):
                if d is not None:
                    pos += d * e
                    if exprHasFunction(d) and not dynamicAllowed:
                        raise Exception('Position variable cannot be a dynamic variable for a joint connection, variable: {}'.format(d))
                    if dynamicAllowed:
                        vel += diff(d,t) * e
            #  Orientation
            if rot_amounts is None:
                raise Exception('rot_amounts needs to be provided with Joint connection')
            for d in rot_amounts:
                if d!=0 and not exprHasFunction(d):
                    raise Exception('Rotation amount variable should be a dynamic variable for a joint connection, variable: {}'.format(d))
        else:
            raise Exception('Unsupported joint type: {}'.format(type))

        # Orientation (creating a path connecting frames together)
        if rot_amounts is None:
            child.frame.orient(parent.frame, 'Axis', (0, parent.frame.x) ) 
        else:
            if rot_type in ['Body','Space']:
                child.frame.orient(parent.frame, rot_type, rot_amounts, rot_order) # <<< 
            else:
                child.frame.orient(parent.frame, rot_type, rot_amounts) # <<< 


        # Position of child origin wrt parent origin
        child.origin.set_pos(parent.origin, pos)
        # Velocity of child origin frame wrt parent frame (0 for rigid or joint)
        child.origin.set_vel(parent.frame, vel);
        # Velocity of child masscenter wrt parent frame, based on origin vel (NOTE: for rigid body only, should be overriden for flexible body)
        child.masscenter.v2pt_theory(child.origin, parent.frame, child.frame);
        # Velocity of child origin wrt inertial frame, using parent origin/frame as intermediate
        child.origin.v1pt_theory(parent.origin, child.inertial_frame, parent.frame)
        # Velocity of child masscenter wrt inertial frame, using parent origin/frame as intermediate
        child.masscenter.v1pt_theory(parent.origin, child.inertial_frame, parent.frame)

        #r_OB = child.origin.pos_from(child.inertial_origin)
        #vel_OB = r_OB.diff(t, child.inertial_frame)

    # --------------------------------------------------------------------------------}
    # --- Visualization 
    # --------------------------------------------------------------------------------{
    
    def vizOrigin(self, radius=1.0, color='black', format='pydy'):
        if format=='pydy':
            from pydy.viz.shapes import Sphere
            from pydy.viz.visualization_frame import VisualizationFrame
            return VisualizationFrame(self.frame, self.origin, Sphere(color=color, radius=radius))

    def vizCOG(self, radius=1.0, color='red', format='pydy'):
        if format=='pydy':
            from pydy.viz.shapes import Sphere
            from pydy.viz.visualization_frame import VisualizationFrame
            return VisualizationFrame(self.frame, self.masscenter, Sphere(color=color, radius=radius))

    def vizFrame(self, radius=0.1, length=1.0, format='pydy'):
        if format=='pydy':
            from pydy.viz.shapes import Cylinder
            from pydy.viz.visualization_frame import VisualizationFrame
            from sympy.physics.mechanics import Point
            X_frame  = self.frame.orientnew('ffx', 'Axis', (-np.pi/2, self.frame.z) ) # Make y be x
            Z_frame  = self.frame.orientnew('ffz', 'Axis', (+np.pi/2, self.frame.x) ) # Make y be z
            X_shape   = Cylinder(radius=radius, length=length, color='red') # Cylinder are along y
            Y_shape   = Cylinder(radius=radius, length=length, color='green')
            Z_shape   = Cylinder(radius=radius, length=length, color='blue')
            X_center=Point('X'); X_center.set_pos(self.origin, length/2 * X_frame.y)
            Y_center=Point('Y'); Y_center.set_pos(self.origin, length/2 * self.frame.y)
            Z_center=Point('Z'); Z_center.set_pos(self.origin, length/2 * Z_frame.y)
            X_viz_frame = VisualizationFrame(X_frame, X_center, X_shape)
            Y_viz_frame = VisualizationFrame(self.frame, Y_center, Y_shape)
            Z_viz_frame = VisualizationFrame(Z_frame, Z_center, Z_shape)
        return X_viz_frame, Y_viz_frame, Z_viz_frame

    def vizAsCylinder(self, radius, length, axis='z', color='blue', offset=0, format='pydy'):
        """ """
        if format=='pydy':
            # pydy cylinder is along y and centered at the middle of the cylinder
            from pydy.viz.shapes import Cylinder
            from pydy.viz.visualization_frame import VisualizationFrame
            if axis=='y':
                e = self.frame
                a = self.frame.y
            elif axis=='z':
                e = self.frame.orientnew('CF_'+self.name, 'Axis', (np.pi/2, self.frame.x) ) 
                a = self.frame.z
            elif axis=='x':
                e = self.frame.orientnew('CF_'+self.name, 'Axis', (np.pi/2, self.frame.z) ) 
                a = self.frame.x

            shape = Cylinder(radius=radius, length=length, color=color)
            center=Point('CC_'+self.name); center.set_pos(self.origin, (length/2 +offset) * a)
            return VisualizationFrame(e, center, shape)
        else:
            raise NotImplementedError()


    def vizAsRotor(self, radius=0.1, length=1, nB=3,  axis='x', color='white', format='pydy'):
        # --- Bodies visualization
        if format=='pydy':
            from pydy.viz.shapes import Cylinder
            from pydy.viz.visualization_frame import VisualizationFrame
            blade_shape = Cylinder(radius=radius, length=length, color=color)
            viz=[]
            if axis=='x':
                for iB in np.arange(nB):
                    frame  = self.frame.orientnew('b', 'Axis', (-np.pi/2+(iB-1)*2*np.pi/nB , self.frame.x) ) # Y pointing along blade
                    center=Point('RB'); 
                    center.set_pos(self.origin, length/2 * frame.y)
                    viz.append( VisualizationFrame(frame, center, blade_shape) )
                return viz
            else:
                raise NotImplementedError()
            
class Body(object):
    def __init__(B,Name=''):
        B.Children    = []
        B.Connections = []
        B.Name        = Name
        B.MM     = None
        B.B           = [] # Velocity transformation matrix
        B.updatePosOrientation(colvec([0,0,0]), eye(3))

    def updatePosOrientation(o,x_0,R_0b):
        o.r_O = x_0      # position of body origin in global coordinates
        o.R_0b=R_0b      # transformation matrix from body to global

    def connectTo(self, Child, Point=None, Type=None, RelOrientation=None, JointRotations=None):
        if Type =='Rigid':
            c=Connection(Type, RelPoint=Point, RelOrientation = RelOrientation)
        else: # TODO first node, last node
            c=Connection(Type, RelPoint=Point, RelOrientation=RelOrientation, JointRotations=JointRotations)
        self.Children.append(Child)
        self.Connections.append(c)

    def setupDOFIndex(o,n):
        nForMe=o.nf
        # Setting my dof index
        o.I_DOF=n+ np.arange(nForMe) 
        # Update
        n=n+nForMe
        for child,conn in zip(o.Children,o.Connections):
            # Connection first
            nForConn=conn.nj;
            conn.I_DOF=n+np.arange(nForConn)
            # Update
            n=n+nForConn;
            # Then Children
            n=child.setupDOFIndex(n)
        return n

    def __repr__(B):
        pass

    @property
    def R_bc(self):
        return eye(3);
    @property
    def Bhat_x_bc(self):
        return Matrix(np.zeros((3,0)))
    @property
    def Bhat_t_bc(self):
        return Matrix(np.zeros((3,0)))

    def updateChildrenKinematicsNonRecursive(p,q):
        # At this stage all the kinematics of the body p are known
        # Useful variables
        R_0p =  p.R_0b
        B_p  =  p.B
        r_0p  = p.r_O

        nf_all_children=sum([child.nf for child in p.Children])

        for ic,(body_i,conn_pi) in enumerate(zip(p.Children,p.Connections)):
            # Flexible influence to connection point
            R_pc  = p.R_bc
            Bx_pc = p.Bhat_x_bc
            Bt_pc = p.Bhat_t_bc
            # Joint influence to next body (R_ci, B_ci)
            conn_pi.updateKinematics(q) # TODO

            # Full connection p and j
            R_pi   = R_pc*conn_pi.R_ci  
            if conn_pi.B_ci.shape[1]>0:
                Bx_pi  = Matrix(np.column_stack((Bx_pc, np.dot(R_pc,conn_pi.B_ci[:3,:]))))
                Bt_pi  = Matrix(np.column_stack((Bt_pc, np.dot(R_pc,conn_pi.B_ci[3:,:]))))
            else:
                Bx_pi  = Bx_pc
                Bt_pi  = Bt_pc
              
            # Rotation of body i is rotation due to p and j
            R_0i = R_0p * R_pi

            # Position of connection point in P and 0 system
            r_pi_inP= conn_pi.s_C_inB
            r_pi    = R_0p * r_pi_inP
            B_i      = fBMatRecursion(B_p, Bx_pi, Bt_pi, R_0p, r_pi)
            B_i_inI  = fB_inB(R_0i, B_i)
            BB_i_inI = fB_aug(B_i_inI, body_i.nf)

            body_i.B      = B_i    
            body_i.B_inB  = B_i_inI
            body_i.BB_inB = BB_i_inI

            # --- Updating Position and orientation of child body 
            r_0i = r_0p + r_pi  # % in 0 system
            body_i.R_pb = R_pi 
            body_i.updatePosOrientation(r_0i,R_0i)

            # TODO flexible dofs and velocities/acceleration
            body_i.gzf  = q[body_i.I_DOF,0] # TODO use updateKinematics

    def getFullM(o,M):
        if not isinstance(o,GroundBody):
            MqB      = fBMB(o.BB_inB,o.MM)
            n        = MqB.shape[0]
            M[:n,:n] = M[:n,:n]+MqB     
        for c in o.Children:
            M=c.getFullM(M)
        return M
        
    def getFullK(o,K):
        if not isinstance(o,GroundBody):
            KqB      = fBMB(o.BB_inB,o.KK)
            n        = KqB.shape[0]
            K[:n,:n] = K[:n,:n]+KqB     
        for c in o.Children:
            K=c.getFullK(K)
        return K
        
    def getFullD(o,D):
        if not isinstance(o,GroundBody):
            DqB      = fBMB(o.BB_inB,o.DD)
            n        = DqB.shape[0]
            D[:n,:n] = D[:n,:n]+DqB     
        for c in o.Children:
            D=c.getFullD(D)
        return D


# --------------------------------------------------------------------------------}
# --- Ground/inertial Body 
# --------------------------------------------------------------------------------{
class YAMSInertialBody(YAMSBody):
    """ Inertial body / ground/ earth 
    Typically only one used
    """
    def __init__(self, name='E'): # "Earth"
        YAMSBody.__init__(self,name)
    

class GroundBody(Body):
    def __init__(B):
        super(GroundBody,B).__init__('Grd')
        B.nf   = 0

# --------------------------------------------------------------------------------}
# --- Rigid Body 
# --------------------------------------------------------------------------------{
class YAMSRigidBody(YAMSBody,SympyRigidBody):
    def __init__(self, name, mass=None, J_G=None, rho_G=None, J_diag=False):
        """
        Define a rigid body and introduce symbols for convenience.
        
           Origin point have no velocities in the body frame! 
            
        
        INPUTS:
            name: string (can be one character), make sure this string is unique between all bodies
        
        OPTIONAL INPUTS:
            mass : scalar, body mass
            J_G  : 3x3 array or 3-array defining the coordinates of the inertia tensor in the body frame at the COG
            rho_G: array-like of length 3 defining the coordinates of the COG in the body frame
            J_diag: if true, the inertial tensor J_G is initialized as diagonal
        
        
        """
        # YAMS Body creates a default "origin", "masscenter", and "frame"
        YAMSBody.__init__(self, name)
        
        # --- Mass
        if mass is None:
            mass=Symbol('M_'+name)
        
        # --- Inertia, creating a dyadic using our frame and G
        if J_G is not None:
            if len(list(J_G))==3:
                ixx=J_G[0]
                iyy=J_G[1]
                izz=J_G[2]
                ixy, iyz, izx =0,0,0
            else:
                J_G = ensureMat(J_G, 3, 3)
                ixx = J_G[0,0]
                iyy = J_G[1,1]
                izz = J_G[2,2]
                izx = J_G[2,0]
                ixy = J_G[0,1]
                iyz = J_G[1,2]
        else:
            ixx = Symbol('J_xx_'+name)
            iyy = Symbol('J_yy_'+name)
            izz = Symbol('J_zz_'+name)
            izx = Symbol('J_zx_'+name)
            ixy = Symbol('J_xy_'+name)
            iyz = Symbol('J_yz_'+name)
        if J_diag:
            ixy, iyz, izx =0,0,0
            
        #inertia: dyadic : (inertia(frame, *list), point)
        _inertia = (inertia(self.frame, ixx, iyy, izz, ixy, iyz, izx), self.masscenter)
            
        # --- Position of COG in body frame
        if rho_G is None: 
            rho_G=symbols('x_G_'+name+ ', y_G_'+name+ ', z_G_'+name)
        self.setGcoord(rho_G)
        self.masscenter.set_vel(self.frame, 0 * self.frame.x)
            
        # Init Sympy Rigid Body 
        SympyRigidBody.__init__(self, name, self.masscenter, self.frame, mass, _inertia)
            
    def inertiaIsInPrincipalAxes(self):
        """ enforce the fact that the frame is along the principal axes"""
        D=self._inertia.to_matrix(self.frame)
        self.inertia=(inertia(self.frame, D[0,0], D[1,1], D[2,2]), self._inertia_point)
            
    def setGcoord(self, rho_G):
        """
        INPUTS:
            rho_G: array-like of length 3 defining the coordinates of the COG in the body frame
        """
        rho_G = ensureList(rho_G,3)
        self.s_G_inB = rho_G[0]*self.frame.x + rho_G[1]*self.frame.y+ rho_G[2]*self.frame.z # coordinates of 
        
        self.masscenter.set_pos(self.origin, self.s_G_inB)
        
    @property    
    def origin_inertia(self):
        return self.parallel_axis(self.origin)
    
    @property    
    def inertia_matrix(self):
        """ Returns inertia matrix in body frame"""
        return self.inertia[0].to_matrix(self.frame)
    
    def __repr__(self):
        s=YAMSBody.__repr__(self)
        s+=' - mass:         {}\n'.format(self.mass)
        s+=' - inertia:      {}\n'.format(self.inertia[0].to_matrix(self.frame))
        s+='   (defined at point {})\n'.format(self.inertia[1])
        s+=' - masscenter:   {}\n'.format(self.masscenter)
        s+='   (position from origin: {})\n'.format(self.masscenter.pos_from(self.origin))
        return s
        
class RigidBody(Body):
    def __init__(B, Name, Mass, J_G, rho_G):
        """
        Creates a rigid body 
        """
        super(RigidBody,B).__init__(Name)
        B.nf  = 0
        B.s_G_inB = rho_G
        B.J_G_inB = J_G  
        B.Mass    = Mass 

# --------------------------------------------------------------------------------}
# --- Flexible body/Beam Body 
# --------------------------------------------------------------------------------{
class YAMSFlexibleBody(YAMSBody):
    def __init__(self, name, nq, directions=None):
        YAMSBody.__init__(self,name)
        self.name=name
        self.L= symbols('L_'+name)
        self.q=[]     # DOF
        self.qd=[]    # DOF velocity as "anonymous" variables
        self.qdot=[]  # DOF velocities
        self.qddot=[] # DOF accelerations
        t=dynamicsymbols._t
        for i in np.arange(nq):
            self.q.append(dynamicsymbols('q_{}{}'.format(name,i+1)))
            self.qd.append(dynamicsymbols('qd_{}{}'.format(name,i+1)))
            self.qdot.append(diff(self.q[i],t))
            self.qddot.append(diff(self.qdot[i],t))
        # --- Mass matrix related
        self.mass=symbols('M_{}'.format(name))
        self.J   = Taylor(self.name,'J'  , 3 , 3, nq=nq, rname='xyz', cname='xyz')
        self.Ct  = Taylor(self.name,'C_t', nq, 3, nq=nq, rname=None , cname='xyz')
        self.Cr  = Taylor(self.name,'C_r', nq, 3, nq=nq, rname=None , cname=['x','y','z'])
        self.Me  = Taylor(self.name,'M_e', nq, nq, nq=nq, rname=None , cname=None)
        self.mdCM= Taylor(self.name,'M_d', 3,  1,  nq=nq, rname='xyz', cname=[''])
        # --- h-omega related terms
        self.Gr = Taylor(self.name, 'G_r', 3,  3,  nq=nq, rname='xyz', cname='xyz')
        self.Ge = Taylor(self.name, 'G_e', nq, 3,  nq=nq, rname=None, cname='xyz')
        self.Oe = Taylor(self.name, 'O_e', nq, 6,  nq=nq, rname=None, cname=['xx','yy','zz','xy','yz','xz'])
        # --- Stiffness and damping
        self.Ke  = Taylor(self.name,'K_e', nq, nq, nq=nq, rname=None , cname=None, order=1)
        self.De  = Taylor(self.name,'D_e', nq, nq, nq=nq, rname=None , cname=None, order=1)
        
        self.defineExtremity(directions)
        
        self.origin = Point('O_'+self.name)
        # Properties from Rigid body
        # inertia
        
        # NOTE: masscenter put at origin for flexible bodies for now
        self.masscenter.set_pos(self.origin, 0*self.frame.x)
 
    def __repr__(self):
        s=YAMSBody.__repr__(self)
        s+=' - mass:         {}\n'.format(self.mass)
        #s+=' - inertia:      {}\n'.format(self.inertia[0].to_matrix(self.frame))
        #s+='   (defined at point {})\n'.format(self.inertia[1])
        #s+=' - masscenter:   {}\n'.format(self.masscenter)
        #s+='   (position from origin: {})\n'.format(self.masscenter.pos_from(self.origin))
        s+=' - q:            {}\n'.format(self.q)
        s+=' - qd:           {}\n'.format(self.qd)
        return s


    def defineExtremity(self, directions=None): 
        if directions is None:
            directions=['xyz']*len(self.q)
        # Hard coding 1 connection at beam extremity
        self.alpha =[dynamicsymbols('alpha_x{}'.format(self.name)),dynamicsymbols('alpha_y{}'.format(self.name)),dynamicsymbols('alpha_z{}'.format(self.name))]
        self.uc    =[dynamicsymbols('u_x{}c'.format(self.name)),dynamicsymbols('u_y{}c'.format(self.name)),0]
        alphax = 0
        alphay = 0
        alphaz = 0
        uxc = 0
        uyc = 0
        vList =[]
        uList =[]
        for i in np.arange(len(self.q)):
            u=0
            v=0
            if 'x' in directions[i]:
                u = symbols('u_x{}{}c'.format(self.name,i+1))
                v = symbols('v_y{}{}c'.format(self.name,i+1)) 
                uxc   += u * self.q[i]
                alphay+= v * self.q[i]
            if 'y' in directions[i]:
                u = symbols('u_y{}{}c'.format(self.name,i+1))
                v = symbols('v_x{}{}c'.format(self.name,i+1))
                uyc   += u * self.q[i]
                alphax+= v * self.q[i]
            if 'z' in directions[i]:
                v = symbols('v_z{}{}c'.format(self.name,i+1))
                alphaz+= v * self.q[i]
            vList.append(v)
            uList.append(u)
        
        self.alphaSubs=[(self.alpha[0], alphax ), (self.alpha[1],alphay), (self.alpha[2],alphaz)]
        self.ucSubs   =[(self.uc[0], uxc ), (self.uc[1],uyc)]
        self.v2Subs = [(v1*v2,0) for v1 in vList for v2 in vList] # Second order terms put to 0

        self.ucList =uList # "PhiU" values at connection point for each mode
        self.vcList =vList # "PhiV" valaes at connection point for each mode
        
    def bodyMassMatrix(self, q=None, form='TaylorExpanded'):
        """ Body mass matrix in body coordinates M'(q)
        """
        nq=len(self.q)
        if q is None:
            q = self.q
        else:
            if len(q)!=len(self.q):
                raise Exception('Inconsistent dimension between q ({}) and body nq ({}) for body {}'.format(len(q),nq,self.name))
        self.M=zeros(6+nq,6+nq)
        # Mxx
        self.M[0,0] = self.mass
        self.M[1,1] = self.mass
        self.M[2,2] = self.mass
        if form=='symbolic':
            # Mxr, Mxg
            for i in np.arange(0,3):
                for j in np.arange(3,6+nq):
                    self.M[i,j]=Symbol('M_{}{}{}'.format(self.name,i+1,j+1))
            self.M[0,3]=0
            self.M[1,4]=0
            self.M[2,5]=0
            # Mrr
            char='xyz'
            for i in np.arange(3,6):
                for j in np.arange(3,6):
                    self.M[i,j]=Symbol('J_{}{}{}'.format(self.name,char[i-3],char[j-3]))
            # Mrg
            for i in np.arange(3,6):
                for j in np.arange(6,6+nq):
                    self.M[i,j]=Symbol('M_{}{}{}'.format(self.name,i+1,j+1))
            # Mgg
            for i in np.arange(6,6+nq):
                for j in np.arange(6,6+nq):
                    self.M[i,j]=Symbol('GM_{}{}{}'.format(self.name,i-5,j-5))

            for i in np.arange(0,6+nq):
                for j in np.arange(0,6+nq):
                    self.M[j,i]=self.M[i,j]
            pass

        elif form=='TaylorExpanded':
            # Mrx, Mxr
            self.M[0:3,3:6] = skew(self.mdCM.eval(q))
            self.M[3:6,0:3] = self.M[0:3,3:6].transpose()
            # Mrr
            self.M[3:6,3:6] = self.J.eval(q)
            # Mgx, Mxg
            self.M[6:6+nq,0:3] = self.Ct.eval(q)
            self.M[0:3,6:6+nq] = self.M[6:6+nq,0:3].transpose()
            # Mrg
            self.M[6:6+nq,3:6] = self.Cr.eval(q)
            self.M[3:6,6:6+nq] = self.M[6:6+nq,3:6].transpose()
            # Mgg
            Mgg = self.Me.eval(q)
            self.M[6:6+nq,6:6+nq] = Mgg
        else:
            raise Exception('Unknown mass matrix form option `{}`'.format(form))
        
        return self.M
    
    def bodyQuadraticForce(self, omega, q, qd):
        """ Body quadratic force  k_\omega (or h_omega)  (centrifugal and gyroscopic)
        inputs:
           omega: angular velocity of the body wrt to the inertial frame, expressed in body coordinates
           q,qd: generalied coordinates and speeds for this body
        """
        # --- Safety
        q     = ensureMat(q , len(q), 1)
        qd    = ensureMat(qd, len(qd), 1)
        omega = ensureMat(omega, 3, 1)
        nq=len(self.q)
        if len(q)!=nq:
            raise Exception('Inconsistent dimension between q ({}) and body nq ({}) for body {}'.format(len(q),nq,self.name))
            
        # --- Init
        k_omega = Matrix(np.zeros((6+nq,1)).astype(int)) 
        om_til = skew(omega)
        ox,oy,oz=omega[0,0],omega[1,0],omega[2,0]
        omega_q = Matrix([ox**2, oy**2,oz**2, ox*oy, oy*oz, ox*oz]).reshape(6,1)
        
        # k_omega_t
        k_omega[0:3,0] = 2 *om_til * transpose(self.Ct.eval(q)) * qd
        k_omega[0:3,0] += om_til * skew(self.mdCM.eval(q)) * omega
        # k_omega_r
        k_omega[3:6,0] = om_til * self.J.eval(q) * omega
        for k in np.arange(nq):
            k_omega[3:6,0] += self.Gr.eval(q) * qd[k] * omega
        # k_omega_e
        k_omega[6:6+nq,0] = self.Oe.eval(q) * omega_q
        for k in np.arange(nq):
            k_omega[6:6+nq,0] += self.Ge.eval(q) * qd[k] * omega

        return k_omega
    
    def bodyElasticForce(self, q, qd):
        # --- Safety
        q     = ensureMat(q , len(q), 1)
        qd    = ensureMat(qd, len(qd), 1)
        nq=len(self.q)
        if len(q)!=nq:
            raise Exception('Inconsistent dimension between q ({}) and body nq ({}) for body {}'.format(len(q),nq,self.name))
        # --- Init
        ke = Matrix(np.zeros((6+nq,1)).astype(int)) 
        ke[6:6+nq,0] = self.Ke.eval(q)*q + self.De.eval(q)*qd
        return ke
    
    def connectTo(parent, child, type, rel_pos, rot_type='Body', rot_amounts=None, rot_order=None, rot_order_elastic='XYZ', rot_type_elastic='SmallRot', doSubs=True):
        """
        The connection between a flexible body and another body is similar to the connections between rigid bodies
        The relative position and rotations between bodies are modified in this function to include the elastic 
        displacements and rotations.
        
        For now, we only allow 1 connection for each flexible body.
        
        rel_pos: for flexible bodies, this is the position when the body is undeflected!
        rot_amounts: for flexible bodies, these are the rotations when the body is undeflected!
                     These rotations are assumed to occur AFTER the rotations from the deflection
                     
        s_PC  : vector from parent origin to child origin
        s_PC0 : vector from parent origin to child origin when undeflected
        
        s_PC = s_PC0 + u
        
        """
        rel_pos = [r + u for r,u in zip(rel_pos, parent.uc)]
        # Computing DCM due to elasti motion
        M_B2e = rotToDCM(rot_type_elastic, rot_amounts = parent.alpha, rot_order=rot_order_elastic) # from parent to deformed parent 
        # Insert elastic DOF
        if doSubs:
            rel_pos = [r.subs(parent.ucSubs) for r in rel_pos]
            M_B2e   = M_B2e.subs(parent.alphaSubs)
        # Full DCM
        if rot_amounts is None:
            M_c2B = M_B2e.transpose()
        else:
            M_e2c = rotToDCM(rot_type,         rot_amounts = rot_amounts,  rot_order=rot_order) # from parent to deformed parent 
            M_c2B = (M_e2c.transpose() * M_B2e.transpose() )
        
        #print('Rel pos with Elastic motion:', rel_pos)
        #print('Rel rot with Elastic motion:', rot_amounts)
            
        YAMSBody.connectTo(parent, child, type, rel_pos=rel_pos, rot_type='DCM', rot_amounts=M_c2B, dynamicAllowed=True)
        #YAMSBody.connectTo(parent, child, type, rel_pos=rel_pos, dynamicAllowed=True)
        
    
class BeamBody(Body):
    def __init__(B,Name,nf,main_axis='z',nD=2):
        super(BeamBody,B).__init__(Name)
        B.nf  = nf
        B.nD  = nD
        B.main_axis = main_axis
    @property
    def alpha_couplings(self):
        return  np.dot(self.Bhat_t_bc , self.gzf)

    @property
    def R_bc(self):
        if self.main_axis=='x':
            alpha_y= symbols('alpha_y') #-p.V(3,iNode);
            alpha_z= symbols('alpha_z') # p.V(2,iNode);
            return R_y(alpha_y)*R_z(alpha_z)

        elif self.main_axis=='z':
            alpha_x= symbols('alpha_x') #-p.V(2,iNode);
            alpha_y= symbols('alpha_y') # p.V(1,iNode);
            return R_x(alpha_x)*R_y(alpha_y)
        else:
            raise NotImplementedError()

    @property
    def Bhat_x_bc(self):
        #      Bx_pc(:,j)=p.PhiU{j}(:,iNode);
        Bhat_x_bc = Matrix(np.zeros((3,self.nf)).astype(int))
        if self.main_axis=='z':
            for j in np.arange(self.nf):
                if j<self.nf/2 or self.nD==1:
                    Bhat_x_bc[0,j]=symbols('ux{:d}c'.format(j+1)) # p.PhiU{j}(:,iNode);  along x
                else:
                    Bhat_x_bc[1,j]=symbols('uy{:d}c'.format(j+1)) # p.PhiU{j}(:,iNode);  along y
        elif self.main_axis=='x':
            for j in np.arange(self.nf):
                if j<self.nf/2 or self.nD==1:
                    Bhat_x_bc[2,j]=symbols('uz{:d}c'.format(j+1)) # p.PhiU{j}(:,iNode);  along z
                else:
                    Bhat_x_bc[1,j]=symbols('uy{:d}c'.format(j+1)) # p.PhiU{j}(:,iNode);  along y
        return Bhat_x_bc

    @property
    def Bhat_t_bc(self):
        #      Bt_pc(:,j)=[0; -p.PhiV{j}(3,iNode); p.PhiV{j}(2,iNode)];
        Bhat_t_bc = Matrix(np.zeros((3,self.nf)).astype(int))
        if self.main_axis=='z':
            for j in np.arange(self.nf):
                if j<self.nf/2  or self.nD==1:
                    Bhat_t_bc[1,j]=symbols('vy{:d}c'.format(j+1))
                else:
                    Bhat_t_bc[0,j]=-symbols('vx{:d}c'.format(j+1))
        elif self.main_axis=='x':
            for j in np.arange(self.nf):
                if j<self.nf/2 or self.nD==1:
                    Bhat_t_bc[1,j]=-symbols('vz{:d}c'.format(j+1))
                else:
                    Bhat_t_bc[2,j]=symbols('vy{:d}c'.format(j+1))
        return Bhat_t_bc




# --------------------------------------------------------------------------------}
# --- Rotation 
# --------------------------------------------------------------------------------{
def R_x(t):
    return Matrix( [[1,0,0], [0,cos(t),-sin(t)], [0,sin(t),cos(t)]])
def R_y(t):
    return Matrix( [[cos(t),0,sin(t)], [0,1,0], [-sin(t),0,cos(t)] ])
def R_z(t): 
    return Matrix( [[cos(t),-sin(t),0], [sin(t),cos(t),0], [0,0,1]])
# --------------------------------------------------------------------------------}
# --- B Matrices 
# --------------------------------------------------------------------------------{
def fB_inB(R_EI, B_I):
    """ Transfer a global B_I matrix (body I at point I) into a matrix in it's own coordinate.
    Simply multiply the top part and bottom part of the B matrix by the 3x3 rotation matrix R_EI
    e.g.
         B_N_inN = [R_EN' * B_N(1:3,:);  R_EN' * B_N(4:6,:)];
    """ 
    if len(B_I)==0:
        B_I_inI = Matrix(np.array([]))
    else:
        B_I_inI = Matrix(np.vstack(( R_EI.T* B_I[:3,:],  R_EI.T * B_I[3:,:])))
    return B_I_inI

def fB_aug(B_I_inI, nf_I, nf_Curr=None, nf_Prev=None):
    """
    Augments the B_I_inI matrix, to include nf_I flexible degrees of freedom.
    This returns the full B matrix on the left side of Eq.(11) from [1], 
    based on the Bx and Bt matrices on the right side of this equation
    """
    if len(B_I_inI)==0:
        if nf_I>0:
            BB_I_inI = Matrix(np.vstack( (np.zeros((6,nf_I)).astype(int), np.eye(nf_I).astype(int))) )
        else:
            BB_I_inI= Matrix(np.zeros((6,0)).astype(int))
    else:
        if nf_Curr is not None:
            # Case of several flexible bodies connected to one point (i.e. blades)
            nf_After=nf_I-nf_Prev-nf_Curr
            I = np.block( [np.zeros((nf_Curr,nf_Prev)), np.eye(nf_Curr), np.zeros((nf_Curr,nf_After))] )
        else:
            nf_Curr=nf_I
            I=np.eye(nf_I)

        BB_I_inI = np.block([ [B_I_inI, np.zeros((6,nf_I))], [np.zeros((nf_Curr,B_I_inI.shape[1])), I]]);

    return Matrix(BB_I_inI)


def fBMatRecursion(Bp, Bhat_x, Bhat_t, R0p, r_pi):
    """ Recursive formulae for B' and Bhat 
    See discussion after Eq.(12) and (15) from [1]
    """
    # --- Safety checks
    if len(Bp)==0:
        n_p = 0
    elif len(Bp.shape)==2:
        n_p = Bp.shape[1]
    else:
        raise Exception('Bp needs to be empty or a 2d array')
    if len(Bhat_x)==0:
        ni = 0
    elif len(Bhat_x.shape)==2:
        ni = Bhat_x.shape[1]
    else:
        raise Exception('Bi needs to be empty or a 2d array')

    r_pi=colvec(r_pi)

    # TODO use Translate here
    Bi = Matrix(np.zeros((6,ni+n_p)))
    for j in range(n_p):
        Bi[:3,j] = Bp[:3,j]+cross(Bp[3:,j],r_pi) # Recursive formula for Bt mentioned after Eq.(15)
        Bi[3:,j] = Bp[3:,j] # Recursive formula for Bx mentioned after Eq.(12)
    if ni>0:
        Bi[:3,n_p:] = R0p*Bhat_x[:,:] # Recursive formula for Bx mentioned after Eq.(15)
        Bi[3:,n_p:] = R0p*Bhat_t[:,:] # Recursive formula for Bt mentioned after Eq.(12)
    return Bi

def fBMatTranslate(Bp,r_pi):
    """
    Rigid translation of a B matrix to another point, i.e. transfer the velocities from a point to another: 
      - translational velocity:  v@J = v@I + om@I x r@IJ
      - rotational velocity   : om@J = om@I
    """
    Bi=np.zeros(Bp.shape)
    if Bp.ndim==1:
        raise NotImplementedError

    for j in range(Bp.shape[1]):
        Bi[0:3,j] = Bp[0:3,j]+np.cross(Bp[3:6,j],r_pi.ravel());
        Bi[3:6,j] = Bp[3:6,j]
    return Bi


def fBMB(BB_I_inI,MM):
    """ Computes the body generalized matrix: B'^t M' B 
    See Eq.(8) of [1] 
    """
    MM_I = np.dot(np.transpose(BB_I_inI), MM).dot(BB_I_inI)
    return MM_I




# --------------------------------------------------------------------------------}
# --- Kane's method 
# --------------------------------------------------------------------------------{
def _initialize_kindiffeq_matrices(coordinates, speeds, kdeqs, uaux=Matrix()):
    """Initialize the kinematic differential equation matrices.
    See sympy.mechanics.kane

    kdeqs: kinematic differential equations
    """
    from sympy import solve_linear_system_LU

    if kdeqs:
        if len(coordinates) != len(kdeqs):
            raise ValueError('There must be an equal number of kinematic differential equations and coordinates.')
        coordinates = Matrix(coordinates)
        kdeqs = Matrix(kdeqs)
        u     = Matrix(speeds)
        qdot  = coordinates.diff(dynamicsymbols._t)
        # Dictionaries setting things to zero
        u_zero = dict((i, 0) for i in u)
        uaux_zero = dict((i, 0) for i in uaux)
        qdot_zero = dict((i, 0) for i in qdot)

        f_k = msubs(kdeqs, u_zero, qdot_zero)
        k_ku = (msubs(kdeqs, qdot_zero) - f_k).jacobian(u)
        k_kqdot = (msubs(kdeqs, u_zero) - f_k).jacobian(qdot)

        f_k = k_kqdot.LUsolve(f_k)
        k_ku = k_kqdot.LUsolve(k_ku)
        k_kqdot = eye(len(qdot))

        _qdot_u_map = solve_linear_system_LU( Matrix([k_kqdot.T, -(k_ku * u + f_k).T]).T, qdot)

        _f_k = msubs(f_k, uaux_zero)
        _k_ku = msubs(k_ku, uaux_zero)
        _k_kqdot = k_kqdot
    else:
        _qdot_u_map = None
        _f_k = Matrix()
        _k_ku = Matrix()
        _k_kqdot = Matrix()
    return _qdot_u_map, _f_k, _k_ku, _k_kqdot



def kane_frstar_alt(bodies, coordinates, speeds, kdeqs, inertial_frame, uaux=Matrix(), udep=None, Ars=None):
    """Form the generalized inertia force."""
    from sympy.core.backend import zeros, Matrix, diff
    from sympy.core.compatibility import range
    from sympy.physics.vector import partial_velocity
    from sympy.physics.mechanics.particle import Particle

    t = dynamicsymbols._t
    N = inertial_frame

    # Derived inputs
    q = Matrix(coordinates) # q
    u = Matrix(speeds) # u
    udot = u.diff(t)
    qdot_u_map,_,_,_ = _initialize_kindiffeq_matrices(q, u, kdeqs, uaux=Matrix())
                                                  
    # Dicts setting things to zero
    udot_zero = dict((i, 0) for i in udot)
    uaux_zero = dict((i, 0) for i in uaux)
    uauxdot   = [diff(i, t) for i in uaux]
    uauxdot_zero = dict((i, 0) for i in uauxdot)
    # Dictionary of q' and q'' to u and u'
    q_ddot_u_map = dict((k.diff(t), v.diff(t)) for (k, v) in qdot_u_map.items())
    q_ddot_u_map.update(qdot_u_map)

    # Fill up the list of partials: format is a list with num elements
    # equal to number of entries in body list. Each of these elements is a
    # list - either of length 1 for the translational components of
    # particles or of length 2 for the translational and rotational
    # components of rigid bodies. The inner most list is the list of
    # partial velocities.
    def get_partial_velocity(body):
        if isinstance(body,YAMSRigidBody) or isinstance(body, SympyRigidBody):
            vlist = [body.masscenter.vel(N), body.frame.ang_vel_in(N)]
        elif isinstance(body, Particle):
            vlist = [body.point.vel(N),]
        elif isinstance(body,YAMSFlexibleBody):
            print('>>>> FlexibleBody TODO, Jv Jo to partials')
            vlist=[body.masscenter.vel(N), body.frame.ang_vel_in(N)]
        else:
            raise TypeError('The body list may only contain either ' 'RigidBody or Particle as list elements.')
        v = [msubs(vel, qdot_u_map) for vel in vlist]
        return partial_velocity(v, u, N)

    partials = [get_partial_velocity(body) for body in bodies]

    # Compute fr_star in two components:
    # fr_star = -(MM*u' + nonMM)
    o = len(u)
    MM = zeros(o, o)
    nonMM = zeros(o, 1)
    zero_uaux      = lambda expr: msubs(expr, uaux_zero)
    zero_udot_uaux = lambda expr: msubs(msubs(expr, udot_zero), uaux_zero)
    for i, body in enumerate(bodies):
        bodyMM = zeros(o, o)
        bodynonMM = zeros(o, 1)
        if isinstance(body,YAMSRigidBody) or isinstance(body, SympyRigidBody):
            # Rigid Body (see sympy.mechanics.kane)
            M     = zero_uaux(       body.mass                )
            I     = zero_uaux(       body.central_inertia     )
            vel   = zero_uaux(       body.masscenter.vel(N)   )
            omega = zero_uaux(       body.frame.ang_vel_in(N) )
            acc   = zero_udot_uaux(  body.masscenter.acc(N)   )
            # --- Mas Matrix
            for j in range(o):
                tmp_vel = zero_uaux(partials[i][0][j])
                tmp_ang = zero_uaux(I & partials[i][1][j])
                for k in range(o):
                    # translational
                    bodyMM[j, k] += M * (tmp_vel & partials[i][0][k])
                    # rotational
                    bodyMM[j, k] += (tmp_ang & partials[i][1][k])
            # --- Full inertial loads Matrix
            inertial_force  = (M.diff(t) * vel + M * acc)
            inertial_torque = zero_uaux((I.dt(body.frame) & omega) + msubs(I & body.frame.ang_acc_in(N), udot_zero) + (omega ^ (I & omega)))  # "&" = dot, "^"=cross
            for j in range(o):
                bodynonMM[j] += inertial_force & partials[i][0][j]
                bodynonMM[j] += inertial_torque & partials[i][1][j]

        elif isinstance(body,YAMSFlexibleBody):
            print('>>>> FlexibleBody TODO')
            M     = zero_uaux(body.mass)
            #I     = zero_uaux(body.central_inertia)
            vel   = zero_uaux(body.origin.vel(N))
            omega = zero_uaux(body.frame.ang_vel_in(N))
            acc   = zero_udot_uaux(body.origin.acc(N))
            inertial_force=0 # Fstar  !<<<< TODO
            inertial_torque=0 # Tstar  !<<<< TODO

        else:
            # Particles
            M = zero_uaux(body.mass)
            vel = zero_uaux(body.point.vel(N))
            acc = zero_udot_uaux(body.point.acc(N))
            inertial_force = (M.diff(t) * vel + M * acc)
            inertial_torque=0 # Tstar
            for j in range(o):
                temp = zero_uaux(partials[i][0][j])
                for k in range(o):
                    bodyMM[j, k] += M * (temp & partials[i][0][k])
                bodynonMM[j] += inertial_force & partials[i][0][j]

        # Perform important substitution and store body contributions
        body.MM_alt     = zero_uaux(msubs(bodyMM, q_ddot_u_map))
        body.nonMM_alt  = msubs(msubs(bodynonMM, q_ddot_u_map), udot_zero, uauxdot_zero, uaux_zero)
        # Cumulative MM and nonMM over all bodies
        MM    += bodyMM
        nonMM += bodynonMM
        # --- Storing for debug
        body.acc_alt             = acc
        body.vel_alt             = vel
        body.omega_alt           = omega
        body.inertial_force_alt  = inertial_force
        body.inertial_torque_alt = inertial_torque
        body.Jv_vect_alt=partials[i][0]
        body.Jo_vect_alt=partials[i][1]
    # End loop on bodies

    # Compose fr_star out of MM and nonMM
    fr_star = -(MM * msubs(Matrix(udot), uauxdot_zero) + nonMM)


    # If there are dependent speeds, we need to find fr_star_tilde
    if udep:
        p = o - len(udep)
        fr_star_ind = fr_star[:p, 0]
        fr_star_dep = fr_star[p:o, 0]
        fr_star = fr_star_ind + (Ars.T * fr_star_dep)
        # Apply the same to MM
        MMi = MM[:p, :]
        MMd = MM[p:o, :]
        MM = MMi + (Ars.T * MMd)

    #self._bodylist = bodies
    #self._frstar = fr_star
    #self._k_d = MM
    #self._f_d = -msubs(self._fr + self._frstar, udot_zero)
    return fr_star, MM

def kane_frstar(bodies, coordinates, speeds, kdeqs, origin, inertial_frame, Omega_Subs=[(None,None)], Mform='TaylorExpanded'):
    """ 
    coordinates "q"
    speeds      "u"
    kdeqs:   relates qdot and udot
    
    """ 
    from sympy.physics.vector import partial_velocity

    nq = len(coordinates)
    MM    = zeros(nq, nq)
    nonMM = zeros(nq, 1)

    O_E = origin
    N = inertial_frame
    t = dynamicsymbols._t

    # Derived inputs
    q = Matrix(coordinates) # q
    u = Matrix(speeds) # u
    udot = u.diff(t)
    qspeeds = q.diff(t)
    qdot_u_map,_,_,_ = _initialize_kindiffeq_matrices(q, u, kdeqs, uaux=Matrix())

    # Dicts setting things to zero
    udot_zero = dict((i, 0) for i in udot)
    qdot_zero = dict((diff(qd,t), 0) for qd in qspeeds)
    # Dictionary of q' and q'' to u and u'
    q_ddot_u_map = dict((k.diff(t), v.diff(t)) for (k, v) in qdot_u_map.items())
    q_ddot_u_map.update(qdot_u_map)

    for i,body in enumerate(bodies):
        bodyMM    = zeros(nq, nq)
        bodynonMM = zeros(nq, 1)
        M     = body.mass
        #print(type(body),isinstance(body, YAMSFlexibleBody), isinstance(body, YAMSRigidBody), isinstance(body, YAMSBody), isinstance(body, SympyBody))
        if isinstance(body, YAMSFlexibleBody):
            P = body.origin
            I = None
        else:
            P = body.masscenter
            I = body.central_inertia

        # --- Step 2: Positions and orientation
        r = P.pos_from(O_E)
        R = N.dcm(body.frame) # from body to inertial

        # --- Step 3/4: Velocities and accelerations
        vel   = P.vel(N)
        omega = body.frame.ang_vel_in(N)
        acc   = P.acc(N)
        alpha = omega.diff(t, N)
        # NOTE: Keep me Alternative: vel from r and omega from identification:
        #vel_drdt = r.diff(t, N).simplify()
        #OmSkew = (R.diff(t) *  R.transpose()).simplify()
        #omega_ident = OmSkew[2,1] * N.x + OmSkew[0,2]*N.y + OmSkew[1,0] * N.z
        #acc_drdt2 = r.diff(t, N).diff(t,N).simplify()

        # --- Step 5 Partial velocities
        # Method 1: use "partial_velocity" function, which returns a vector
        vel_sub   = msubs(vel, qdot_u_map)
        Jv_vect   = partial_velocity([vel_sub], u, N)[0]
        omega_sub = msubs(omega, qdot_u_map)
        Jo_vect = partial_velocity([omega_sub], u, N)[0]
        # NOTE: Keep me: Method 2: express everything in ref frame, and use "jacobians"
        #v  = vel.subs(Omega_Subs).to_matrix(N)
        #om = omega.subs(Omega_Subs).to_matrix(N)
        #Jv = v.jacobian(qspeeds)
        #Jo = om.jacobian(qspeeds)

        # --- Step 6 Inertia forces
        if isinstance(body,YAMSRigidBody) or isinstance(body, SympyRigidBody):
            # --- Mass Matrix 
            for j in range(nq):
                tmp_vel = Jv_vect[j]      # Jv[:,j]
                tmp_ang = I & Jo_vect[j]  # Jo[:,j]
                for k in range(nq):
                    # translational
                    bodyMM[j, k] += M * (tmp_vel & Jv_vect[k]) # M * Jv[:,j] dot Jv[:,k]
                    # rotational
                    bodyMM[j, k] +=     (tmp_ang & Jo_vect[k]) # I dot Jo[:,j] dot Jo[:,k]

            # --- Full inertial loads
            inertial_force  = (M.diff(t) * vel + M * acc) # "Fstar"
            inertial_torque = (I.dt(body.frame) & omega) + msubs(I & body.frame.ang_acc_in(N), udot_zero) + (omega ^ (I & omega))  # "&" = dot, "^"=cross

            # NOTE KEEP ME: Alternative formulation using "matrices" 
            #inertial_force = inertial_force.subs(Omega_Subs)
            #RIRt  = R*I.to_matrix(body.frame)*R.transpose()
            #inertial_torque_2 = - RIRt * alpha.to_matrix(N) \
            #        - coord2vec(om, N).cross( coord2vec(RIRt *om, N)).to_matrix(N)

            # Computing generatlized force Jv.f + Jo*M
            for j in range(nq):
                bodynonMM[j] += inertial_force  & Jv_vect[j]
                bodynonMM[j] += inertial_torque & Jo_vect[j]
            # Alternative:
            #frstar_t+ = Jv.transpose() * inertial_force.to_matrix(N)
            #frstar_o+ = Jo.transpose() * inertial_torque

        elif isinstance(body,YAMSFlexibleBody):
            MMloc = body.bodyMassMatrix(form=Mform)
            print('>>>> FlexibleBody TODO')
            inertial_force=0 # Fstar
            inertial_torque=0 # Tstar
            Tstar=0

        else:
            raise Exception('Unsupported body type: {}'.format(type(body)))

        # Perform important substitution and store body contributions
        body.MM      = msubs(bodyMM, q_ddot_u_map)
        body.nonMM   = msubs(msubs(bodynonMM, q_ddot_u_map), udot_zero) #, uauxdot_zero, uaux_zero)

        # Cumulative MM and nonMM over all bodies
        MM   +=bodyMM
        nonMM+=bodynonMM
        # --- Storing for debug
        body.acc             = acc
        body.vel             = vel
        body.omega           = omega
        body.inertial_force  = inertial_force
        body.inertial_torque = inertial_torque
        body.Jv_vect=Jv_vect
        body.Jo_vect=Jo_vect
    # End loop on bodies

    # Compose fr_star out of MM and nonMM
    fr_star = -(MM *Matrix(udot) + nonMM)


    return fr_star, MM

# --------------------------------------------------------------------------------}
# --- Kane fr 
# --------------------------------------------------------------------------------{
def kane_fr_alt(loads, coordinates, speeds, kdeqs, inertial_frame, uaux=Matrix(), udep=None):
    """
      - For each force: compute the velocity at the point of application, v_P, and then do
        fr =  [d v/ dqdot]^t F
      - For each moment: compute the angular velocity of the frame (in E)
        fr =  [d om/ dqdot]^t M
    """
    from sympy.physics.vector import partial_velocity
 

    def _f_list_parser(fl, ref_frame):
        """Parses the provided forcelist composed of items of the form (obj, force).
        Returns a tuple containing:
            vel_list: The velocity (ang_vel for Frames, vel for Points) in the provided reference frame.
            f_list: The forces.
        Used internally in the KanesMethod and LagrangesMethod classes.
        """
        def flist_iter():
            for pair in fl:
                obj, force = pair
                if isinstance(obj, ReferenceFrame):
                    yield obj.ang_vel_in(ref_frame), force
                elif isinstance(obj, Point):
                    yield obj.vel(ref_frame), force
                else:
                    raise TypeError('First entry in each forcelist pair must be a point or frame.')
        if not fl:
            vel_list, f_list = (), ()
        else:
            unzip = lambda l: list(zip(*l)) if l[0] else [(), ()]
            vel_list, f_list = unzip(list(flist_iter()))
        return vel_list, f_list

    """
    Form the generalized active force.
        See _form_fr in sympy.mechanics
    """
    N = inertial_frame
    # Derived inputs
    speeds = Matrix(speeds) # u
    qdot_u_map,_,_,_ = _initialize_kindiffeq_matrices(coordinates, speeds, kdeqs, uaux=Matrix())
    
    # pull out relevant velocities for constructing partial velocities
    vel_list, f_list = _f_list_parser(loads, N)
    vel_list = [msubs(i, qdot_u_map) for i in vel_list]
    f_list   = [msubs(i, qdot_u_map) for i in f_list]
    print(vel_list)

    # Fill Fr with dot product of partial velocities and forces
    o = len(speeds)
    b = len(f_list)
    FR = zeros(o, 1)
    partials = partial_velocity(vel_list, speeds, N)
    #print('>>> ', len(partials), len(partials[0]))
    for i in range(o):
        FR[i] = sum(partials[j][i] & f_list[j] for j in range(b))

    # In case there are dependent speeds
    if udep:
        p = o - len(udep)
        FRtilde = FR[:p, 0]
        FRold = FR[p:o, 0]
        FRtilde += Ars.T * FRold
        FR = FRtilde
    #self._forcelist = loads
    #self._fr = FR
    return FR


def kane_fr(body_loads, speeds, inertial_frame):
    """
    Compute Kane's "fr" terms, using a list of bodies and external loads

    For each body:  fr = Jv * F@refP  + Jo * M
    where Jv and Jo are the jacoban of the linear velocity of the point (for a force) and angular velocity of the frame (for a moment). The point "refP" should be the one used in the calculation of Jv (typically the center of mass for rigid body)

    Right now, Jv, and Jo are computed when calling kane_frstar...

    Alternatively (see kane_fr_alt and _form_fr):
      - For each force: compute the velocity at the point of application, v_P, and then do
        fr =  [d v/ dqdot]^t F
      - For each moment: compute the angular velocity of the frame (in E)
        fr =  [d om/ dqdot]^t M

    INPUTS:
        body_loads: a list of tuples of the form  (body, (point_or_frame, force_or_moment ) )
            The tuples (point_or_frame, force_or_moment) are the ones needed when calling sympy's kane
            For instance:
               body_loads = [
                    (nac, (N        , Thrust*N.x)),
                    (nac, (nac.frame, Qaero*N.x ))
                    ]
        
    """
    import sympy.physics.vector as vect

    fr_t = zeros(len(speeds), 1)
    fr_o = zeros(len(speeds), 1)

    N = inertial_frame

    for bl in body_loads:
        body, (point_or_frame, force_or_moment) = bl
        if not hasattr(body,'Jv_vect') or not hasattr(body, 'Jo_vect'):
            raise Exception('Jacobians matrices need to be computed for body {}. (Call frstart first)'.format(body.name))
        if isinstance(point_or_frame, ReferenceFrame):
            # Moment and frame
            Moment = force_or_moment
            for j in range(len(speeds)):
                fr_o[j] += body.Jo_vect[j] & Moment # Jo^t * M
            pass
        else:
            # Force and point
            Force = force_or_moment
            point = point_or_frame
            r = point.pos_from(body.masscenter)
            for j in range(len(speeds)):
                fr_t[j] += body.Jv_vect[j] & Force  # Jv^t * F
            # Need to add moment if r/=0
            for j in range(len(speeds)):
                fr_o[j] += body.Jo_vect[j] & (vect.cross( r, Force)) # Jo^t * M
    return fr_t+fr_o
