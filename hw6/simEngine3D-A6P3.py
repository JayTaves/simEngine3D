from gcons import *

cons = ReadModelFile('hw6/hw6.mdl')
dp1 = cons[0]
cd = cons[1]

print(dp1.GetPhi())
print(dp1.GetNu())
print(dp1.GetGamma())
print(dp1.GetPhiR())
print(dp1.GetPhiP())

print(cd.GetPhi())
print(cd.GetNu())
print(cd.GetGamma())
print(cd.GetPhiR())
print(cd.GetPhiP())
