"""
Classes for use in simulating state preparation in CeNTREX
"""

#Import necessary packages:
import sys
import pickle
import numpy as np
import scipy

#Custom classes for defining molecular states and some convenience functions for them
sys.path.append('./molecular-state-classes-and-functions/')
from classes import UncoupledBasisState, CoupledBasisState, State
from functions import (make_hamiltonian, make_hamiltonian_B, make_QN, ni_range, vector_to_state,
                        find_state_idx_from_state, make_transform_matrix, matrix_to_states,
                        find_closest_state, make_H_mu)
from matrix_element_functions import ED_ME_mixed_state_uc

class MicrowaveField:
    """
    Define a class for microwave fields.

    inputs:
    Omega_peak = peak Rabi rate [2*pi*Hz]
    p_t = polarization of field function of time (lambda function of t)
    Jg = J for ground state
    Je = J for excited (higher energy) state
    ground_main = ground_state for the main transition being driven (used for calculating Omega
                  and detuning)
    excited_main = ground_state for the main transition being driven (used for calculating Omega
                   and detuning)
    detuning = detuning of field from resonance with the main transition
    Omega_R = Rabi rate profile as function of time normalized so that Omega_R = 1 when at peak Rabi
              rate (lambda function of t)
    Z0 = Z-position where the peak in the intensity distribution is located [m]
    p_main = polarization used to calculate matrix element for main transition
    """
    #Initialization
    def __init__(self, Omega_peak, p_t, Jg, Je, ground_main = None, excited_main = None,
                 Omega_r = None, z0 = 0., detuning = 0, p_main = None):
        self.Omega_peak = Omega_peak
        self.p_t = p_t
        self.Jg = Jg
        self.Je = Je

        #By default, the states used as the main transition are taken to be the singlet states
        if ground_main is None:
            self.ground_main =((1*UncoupledBasisState(J = Jg, mJ = 0,I1 = 1/2, m1 = 1/2, I2 = 1/2, m2 = -1/2)
                                -1*UncoupledBasisState(J = Jg, mJ = 0,I1 = 1/2, m1 = -1/2, I2 = 1/2, m2 = 1/2))
                                /np.sqrt(2))
        else:
            self.ground_main = ground_main

        if excited_main is None:
            self.excited_main = ((1*UncoupledBasisState(J = Je, mJ = 0,I1 = 1/2, m1 = 1/2, I2 = 1/2, m2 = -1/2)
                                 -1*UncoupledBasisState(J = Je, mJ = 0,I1 = 1/2, m1 = -1/2, I2 = 1/2, m2 = 1/2))
                                 /np.sqrt(2))
        else:
            self.excited_main = excited_main
        
        
        self.detuning = detuning
        if Omega_r is None:
            self.Omega_r = lambda t: Omega_peak
        else:
            self.Omega_r = Omega_r

        self.z0 = z0
        self.p_main = p_main

    #Method for finding the exact eigenstates of a given Hamiltonian that most closely correspond
    #to the excited and ground states defined for the field
    def find_closest_eigenstates(self, H, QN):
        """
        inputs:
        H = Hamiltonian whose eigenstates are to be used
        QN =  the basis in which the Hamiltonian is expressed
        """
        ground_main = self.ground_main
        excited_main = self.excited_main

        #Find the eigenstates that correspond to the "main" transition
        ground_main = find_closest_state(H, ground_main, QN)
        excited_main = find_closest_state(H, excited_main, QN)

        #Redefine the states of the field
        self.ground_main = ground_main
        self.excited_main = excited_main

    #Method for calculating angular part of matrix element for main transition
    def calculate_ME_main(self, pol_vec = np.array([0,0,1])):
       #If polarization to be used for main transition was specified, use it
        if self.p_main is not None:
            pol_vec = self.p_main
        
        ME_main = ED_ME_mixed_state_uc(self.excited_main, self.ground_main, pol_vec = pol_vec)
        return ME_main

    #Method to generate matrices describing couplings due to the field
    def generate_couplings(self, QN):
        """
        inputs:
        QN = list of quantum numbers that define the basis for the coupling Hamiltonian
        """
        Jg = self.Jg
        Je = self.Je

        #Loop over possible polarizations and generate coupling matrices
        H_list = []
        for i in range(0,3):
            #Generate polarization vector
            pol_vec = np.array([0,0,0])
            pol_vec[i] = 1

            #Generate coupling matrix
            H = make_H_mu(Jg, Je, QN, pol_vec = pol_vec)
            
            #Remove small components
            H[np.abs(H) < 1e-3*np.max(np.abs(H))] = 0

            #Check that matrix is Hermitian
            is_hermitian = np.allclose(H,H.conj().T)
            if not is_hermitian:
                print("Warning: Microwave coupling matrix {} is not Hermitian!".format(i))

            #Convert to sparse matrix
            # H = csr_matrix(H)

            #Append to list of coupling matrices
            H_list.append(H)

        return H_list

    #Method for calculating the transition frequency (in 2pi*Hz)
    def calculate_frequency(self, H, QN):
        ground_main = self.ground_main
        excited_main = self.excited_main
        delta = self.detuning

        #Diagonalize the Hamiltonian
        D, V = np.linalg.eigh(H)

        
        i_g = find_state_idx_from_state(H,ground_main, QN)
        i_e = find_state_idx_from_state(H,excited_main, QN)

        #Find the transition frequency for main transition
        omega0 = np.real(D[i_e] - D[i_g])

        #Calculate shift (defining detuning = omega-omega0)
        omega = omega0

        return omega

    #Method for generating the diagonal matrix that shifts energies in rotating frame
    def generate_D(self, omega0, H, QN):
        Je = self.Je
        delta = self.detuning

        #Calculate shift (defining detuning = omega-omega0)
        omega = omega0 + delta

        #Generate the shift matrix
        D = np.zeros(H.shape)
        for i in range(0, len(D)):
            if QN[i].J == Je:
                D[i,i] = -omega

        return D

    #Method for finding the time-dependence of Omega based on given Rabi rate profile
    def find_Omega_t(self, r_t):
        Omega_peak = self.Omega_peak
        Omega_r = self.Omega_r
        Omega_t = lambda t: Omega_peak*Omega_r(r_t(t))
        
        return Omega_t