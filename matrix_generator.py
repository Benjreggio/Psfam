import numpy as np
import galois as gf
GF2 = gf.GF(2)

def companion_matrix(m):
  field = gf.GF(2**m)
  a = list(field.irreducible_poly.coeffs[1:])
  C=[]
  for i in range(m-1):
    row = [GF2(0)]*m
    row[i+1] = GF2(1)
    C = C + [row]
  lastrow = []
  for i in range(m):
    lastrow = lastrow + [a[m-i-1]]
  C = C + [lastrow]
  return C

def get_b(a):
  m=len(a)
  b = [GF2(1)]
  for i in range(1,m):
    bn = GF2(0)
    for k in range(1,i-1):
      bn = bn + a[m-i+k]*b[k]
    b = b + [bn]
  return b

def friend_matrix(C):
  m = len(C)
  a = C[m-1]
  b=get_b(a)
  firstrow = [GF2(0)]*m
  firstrow[0] = GF2(1)
  F = []
  for i in range(0,m):
    nextrow = [GF2(0)]*m
    nextrow[m-1-i] = GF2(1)
    for j in range(1,i+1):
      nextrow[m-1-i + j] =  a[j]
    F = F + [nextrow]
  return F

def invert(C):
  m = len(C)
  A=[]
  for i in range(m):
    row = [GF2(0)]*m
    row[i] = GF2(1)
    A = A + [C[i] + row]
  
  for i in range(m):
    j=i+1
    while(A[i][i] != GF2(1)):
      if A[j][i] == GF2(1):
        t = A[j].copy()
        A[j] = A[i].copy()
        A[i] = t
      j=j+1

    for j in range(i+1,m):
      if A[j][i] == GF2(1):
        for k in range(2*m):
          A[j][k] = A[i][k] + A[j][k]

  for i in range(m):
    for j in range(i+1,m):
      if A[i][j] == 1:
        for k in range(2*m):
          A[i][k] = A[i][k] + A[j][k]
  
  B=[]
  for i in range(m):
    B = B + [A[i][m:]]
  return B

def split(f):
  m = len(f)
  flippers = []
  base = -1
  for i in range(m):
    if(f[i][i] == GF2(1)):
      if base == -1:
        base = i
    else:
      flippers = flippers + [i]
  N = []
  #print("Base = " + str(base))
  for i in range(m):
    nextrow = [GF2(0)]*m
    nextrow[i] = GF2(1)
    if(i == base):
      for j in flippers:
        nextrow[j] = GF2(1)
    N = N + [nextrow]
  return N

def transpose(M):
  M2 = []
  for i in range(len(M)):
    row = []
    for j in range(len(M)):
      row = row + [M[j][i]]
    M2 = M2 + [row]
  return M2

def mtimes(M1,M2):
  if(type(M2[0]) == list):
    M3 = []
    m = len(M1)
    for i in range(m):
      row = []
      for j in range(m):
        s=GF2(0)
        for k in range(m):
          s = s + M1[i][k]*M2[k][j]
        row = row + [s]
      M3 = M3 + [row]
    return M3
  else:
    r = []
    m = len(M1)
    for j in range(m):
      s=GF2(0)
      for k in range(m):
          s = s + M1[j][k]*M2[k]
      r = r + [s]
    return r

def is_invertible(B):
  m = len(B)
  A = []
  for i in range(m):
    row = [GF2(0)]*m
    row[i] = GF2(1)
    A = A + [B[i].copy() + row]
  
  for i in range(m):
    j=i+1
    while(A[i][i] != GF2(1)):
      if j>=m:
        return False
      if A[j][i] == GF2(1):
        t = A[j].copy()
        A[j] = A[i].copy()
        A[i] = t
      j=j+1


    for l in range(i+1,m):
      if A[l][i] == GF2(1):
        for k in range(2*m):
          A[l][k] = A[i][k] + A[l][k]

  return True


def split3(M):
  n = len(M)
  if(n == 1):
    return [[GF2(1)]]
  for i in range(n):
    M2 = []
    for j in range(n):
      if(j != i):
        nextrow = M[j].copy()
        nextrow.pop(i)
        M2 = M2 + [nextrow]

    if(is_invertible(M2)):

      L = split3(M2)
      if(L):

        if(is_invertible(L)):
          L_prime = invert(L)

          b = M[i].copy()
          b.remove(b[i])
          eta = mtimes(L_prime,b)
          LR = []
          for j in range(n):
            nextrow = []
            if (j<i):
              nextrow = L[j][0:i] + [GF2(0)] + L[j][i:n-1]
            if(j == i):
              nextrow = eta[0:i] + [GF2(1)] + eta[i:n-1]
            if (j>i):
              nextrow = L[j-1][0:i] + [GF2(0)] + L[j-1][i:n-1]
            LR = LR + [nextrow]
          return LR
  return False


def get_generating_matrix(m):
    C = companion_matrix(m)
    F = friend_matrix(C)
    N = split(F)
    M = mtimes(transpose(N),mtimes(F,N))
    LN = split3(M)

    R=mtimes(transpose(LN),invert(N))
    NF = mtimes(transpose(R),R)

    A=mtimes(R,mtimes(C,invert(R)))
    for i in range(m):
        for j in range(m):
            if(A[i][j] == GF2(1)):
                A[i][j] = 1
            else:
                A[i][j] = 0
    return A