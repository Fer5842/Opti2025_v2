from gurobipy import Model, GRB, quicksum
import numpy as np
import pandas as pd

# --------Carga de datos----------------------------
relaves_df = pd.read_csv("datos/Relaves.csv")
cagua_df = pd.read_csv("datos/Cagua.csv")
bmax_df = pd.read_csv("datos/Bmax.csv")
p_df = pd.read_csv("datos/P.csv")
pmax_df = pd.read_csv("datos/Pmax.csv")
uagua_df = pd.read_csv("datos/Uagua.csv")
cveginst_df = pd.read_csv("datos/Cveginst.csv")
hcubierta_df = pd.read_csv("datos/Hcubierta.csv")
cvegmant_df = pd.read_csv("datos/Cvegmant.csv")
hmant_df = pd.read_csv("datos/Hmant.csv")
pmprom_df = pd.read_csv("datos/PMprom.csv")
pmmax_df = pd.read_csv("datos/PMmax.csv")

pcubierta_df = pd.read_csv("datos/Pcubierta.csv")
fuentes_df = pd.read_csv("datos/Fuentes.csv")
arcos_df = pd.read_csv("datos/Arcos.csv")
pmbase_df = pd.read_csv("datos/PMbase.csv")
wbase_df = pd.read_csv("datos/Wbase.csv")
cflujo_df = pd.read_csv("datos/Cflujo.csv")
kflujo_df = pd.read_csv("datos/Kflujo.csv")

#FALTAN
#pmapagregado_df = pd.read_csv("datos/PMapagregado.csv")
#alpha_df = pd.read_csv("datos/Alpha.csv")
#beta_df = pd.read_csv("datos/Beta.csv")
#a_df = pd.read_csv("datos/A.csv")
#wentrante_df = pd.read_csv("datos/Wentrante.csv")

# --------Datos--------------------------------------

# --------Conjuntos..--------------------------------
#meses de planificación, t
T = list(range(12))  # 12 meses
#relaves, r
R = relaves_df["Relave"].tolist()  #5r['Talabre', 'Pampa Austral', 'Potrerillos II', 'Ovejeria', 'Caren']
#fuentes de agua, f 
F = fuentes_df["Fuentes"].tolist()
#nodos de la red, fuentes, nodos intermedios y relaves, n
N = F + R  
#arcos de la red, (i,j)
A = [(row["Fuentes"], row["Relaves"]) for _, row in arcos_df.iterrows()]

# --------Parámetros----------------------------------
#costo por ton de agua aplicado en relave r, mes t
C_agua = {(row["Relave"], row["Mes"]-1): row["Cagua"] for _, row in cagua_df.iterrows()} #diccionario
#costo instalar cubierta vegetal en relave r, mes t
C_veginst = {(row["Relave"], row["Mes"]-1): row["Cveginst"] for _, row in cveginst_df.iterrows()}
#presupuesto maximo a lo largo del periodo de planificación
B_max = int(bmax_df["Bmax"].iloc[0])
#periodos de duración de permanencia de cubierta vegetal en relave r 
P = {row["Relave"]: int(row["P"]) for _, row in p_df.iterrows()}
#cantidad maxima de periodos consecutivos que pueden pasar sin hacer mantención a un relave
P_max = {row["Relave"]: int(row["Pmax"]) for _, row in pmax_df.iterrows()}
#cantidad agua maxima que se puede aplicar al relave r en cualquier periodo sin considerar cubierta vegetal
U_agua = {row["Relave"]: int(row["Uagua"]) for _, row in uagua_df.iterrows()}
#agua minima necesaria para humedecer una cubierta vegetal a su nivel optimo en relave r por mes de planificación
H_cubierta = {row["Relave"]: row["Hcubierta"] for _, row in hcubierta_df.iterrows()}
#costo mantención cubierta vegetal en relave r, mes t
C_vegmant = {(row["Relaves"], row["Mes"]-1): row["Cvegmant"] for _, row in cvegmant_df.iterrows()}
#umbral maximo de PM permitido en relave r por mes de planificación t, el csv esta anual se divide en 12
PM_max = {(row["Relaves"], t): row["PMmax"]/12 for _, row in pmmax_df.iterrows() for t in T}

#umbral promedio maximo de PM permitido para todos los relaves r en1 mes
#ojo Pm_prom debe esar ug/m^3 y esta en ton/mes en csv, por lo que 1ton= 1x10^9 ug, 
PM_prom = {t: float(pmprom_df["PMprom"].iloc[0]) for t in T} #un único valor promedio que se aplica a todos los meses.

#agua minima necesaria para humedecer relave r cuando se realiza mantención por mes de planificación
H_mant = {row["Relaves"]: row["Hmant"] /1000 for _, row in hmant_df.iterrows()}
#H_mant = {r: 50 for r in R}

#cantidad  maxima de periodos consecutivos que pueden pasar sin satisfacer completamente el requerimiento de agua de la cubierta
P_cubierta = {row["Relaves"]: int(row["Pcubierta"]) for _, row in pcubierta_df.iterrows()}
#concentracion inicial de PM en relave r
#PM_base = {r: 10 for r in R}
PM_base = {row["Relaves"]: float(row["PMbase"]) for _, row in pmbase_df.iterrows()}
#cantidad agua inicial en fuente f
#W_base = {f: 1000 for f in F}
W_base = {row["Fuentes"]: float(row["Wbase"]) for _, row in wbase_df.iterrows()}
#costo transportar 1 ton de agua por arco (i.j)
#C_flujo = {(i,j): 5 for (i,j) in A}  # todos los arcos tienen costo 5
C_flujo = {(row["Fuentes"], row["Relaves"]): float(row["Cflujo"]) for _, row in cflujo_df.iterrows()}
#capacidad maxima de flujo de agua en argo (i,j) mensualmente
#K_flujo ={(i,j): 500 for (i,j) in A}
K_flujo = {(row["Fuentes"], row["Relaves"]): float(row["Kflujo"]) for _, row in kflujo_df.iterrows()}

#------FALTAN:---------
#concentracion de PM que se agrega al relave r, mes t
PM_agregado = {(r,t): 2 for r in R for t in T}
#reducción PM por cada ton de agua aplicada en relave r
alpha = {r: 0.1 for r in R}
#maxima reducción PM por cubierta vegetal en relave r por mes de planificación
beta = {r: 5 for r in R}
#ton de agua minima que se aplica a un relave r si se decide humedecerlo
a = {r: 50 for r in R}
#flujo neto entrante a fuente f durante mes t.
W_entrante = {(f,t): 100 for f in F for t in T}

# --------Generar el modelo-----------------------------
model = Model()

# --------Variables-------------------------------------
x_agua = model.addVars(R, T, vtype=GRB.CONTINUOUS, name="x_agua", lb=0)              
x_cubierta = model.addVars(R, T, vtype=GRB.CONTINUOUS, name="x_cubierta", lb=0)      
x_flujo = model.addVars(A, T, vtype=GRB.CONTINUOUS, name="x_flujo", lb=0)           

y_veg = model.addVars(R, T, vtype=GRB.BINARY, name="y_veg")
y_mant = model.addVars(R, T, vtype=GRB.BINARY, name="y_mant")
y_agua = model.addVars(R, T, vtype=GRB.BINARY, name="y_agua")

z_veg = model.addVars(R, T, vtype=GRB.BINARY, name="z_veg")

rr = model.addVars(R, T, vtype=GRB.BINARY, name="rr")

PM = model.addVars(R, T, vtype=GRB.CONTINUOUS, name="PM", lb=0)               
W = model.addVars(F, T, vtype=GRB.CONTINUOUS, name="W", lb=0)                 
beta_efectivo = model.addVars(R, T, vtype=GRB.CONTINUOUS, name="beta_efectivo", lb=0)        

# ---------Llamamos a update------------------------------
model.update()

# ---------Restricciones-----------------------------------
#R1 Consistencia de agua
for r in R:
    for t in T:
        model.addConstr(a[r]*y_agua[r,t] <= x_agua[r,t])
        model.addConstr(x_agua[r,t] <= U_agua[r]*y_agua[r,t])

#R2 Capacidad de cubierta
for r in R:
    for t in T:
        model.addConstr(x_cubierta[r,t] <= H_cubierta[r]*z_veg[r,t])

#R3 Restistencia de cubierta - REVISAR
for r in R:
    for t in T:
        model.addConstr(quicksum(1 - rr[r,t2] for t2 in range(t, min(t + P_cubierta[r], len(T)))) <= (P_cubierta[r] * z_veg[r,t]) + (len(T) * (1 - z_veg[r,t])))
        #model.addConstr(quicksum(1-rr[r,t2] for t2 in range(t,min(t+P_cubierta,len(T)))) <= (P_cubierta*z_veg[r,t])+(len(T)*(1 - z_veg[r,t])))

#R4 Reducción de PM efectiva
for r in R:
    for t in T:
        model.addConstr(beta_efectivo[r,t] <= (x_cubierta[r,t]/H_cubierta[r])*beta[r])
        model.addConstr(beta_efectivo[r,t] <= z_veg[r,t]*beta[r])

#R5 Flujo de la concentración de material particulado
for r in R:
    for t in T:
        #mes=1
        if t == 0:
            model.addConstr(PM_base[r] + PM_agregado[r,t] - alpha[r]*(x_agua[r,t] - x_cubierta[r,t]) - beta_efectivo[r,t] == PM[r,t])
        #meses posteriores
        if t > 0: #como empezamos en t=0, t=1 es el 2do mes
            model.addConstr(PM[r,t-1] + PM_agregado[r,t] - alpha[r]*(x_agua[r,t] - x_cubierta[r,t]) - beta_efectivo[r,t] == PM[r,t])

#R6 Control de PM por relave
for r in R:
    for t in T:
        model.addConstr(PM[r,t] <= PM_max[r,t])

#R7 Control de PM promedio
for t in T:
    model.addConstr((1/len(R))*quicksum(PM[r,t] for r in R) <= PM_prom[t])

#R8 Agua mínima si hay mantención
for r in R:
    for t in T:
        model.addConstr(x_agua[r,t] >= H_mant[r]*y_mant[r,t])

#R9 Ventana de mantención obligatoria
for r in R:
    for t in T:
        model.addConstr(quicksum(y_mant[r, t2] for t2 in range(max(0, t-P_max[r]+1), t+1)) >= 1)

#R10 Acciones con mantención activa
for r in R:
    for t in T:
        model.addConstr(y_veg[r,t] <= y_mant[r,t])

#R11 Presencia de la cubierta vegetal
'''for r in R:
    for t in T:
        model.addConstrs((z_veg[r,t] >= y_veg[r,t2] for t2 in range(max(0, t - P[r] + 1), t + 1)))
        model.addConstr(z_veg[r,t] <= quicksum(y_veg[r,t2] for t2 in range(max(0, t - P[r] + 1), t + 1)))
        model.addConstr(quicksum(y_veg[r,t2] for t2 in range(t, min(t + P[r], len(T)))) <= 1)
'''
for r in R:
    for t in T:
        model.addConstr(z_veg[r,t] >= y_veg[r,t])
        model.addConstr(z_veg[r,t] <= quicksum(y_veg[r,t2] for t2 in range(max(0, t-P[r]+1), t+1)))
        if t > 0:
            model.addConstr(z_veg[r,t] <= z_veg[r,t-1] + y_veg[r,t])
            model.addConstr(z_veg[r,t-1] + y_veg[r,t] <= 1)

#R12 Presupuesto total
model.addConstr(
    quicksum(C_agua[r,t]*x_agua[r,t] + C_vegmant[r,t]*z_veg[r,t] + C_veginst[r,t]*y_veg[r,t] for r in R for t in T) 
    + quicksum(C_flujo[i,j]*x_flujo[i,j,t] for (i,j) in A for t in T)
    <= B_max
)
#Se elimino R13 Balance de flujo en nodos intermedios

#R14 Condición inicial de la fuente
for f in F:
    model.addConstr(W[f,0] == W_base[f])

#R15 Balance de la fuente para cada mes t 
for f in F:
    for t in T:
        if t > 0:
            model.addConstr(W[f,t] == W[f,t-1] + W_entrante[f,t] - quicksum(x_flujo[i,j,t] for (i,j) in A if i == f))

#Se elimino R16 No extraer más de lo disponible

#R17 Nodos de relave
for r in R:
    for t in T:
        model.addConstr(quicksum(x_flujo[i,j,t] for (i,j) in A if j == r) == x_agua[r,t])

#R18 Capacidad de las tuberías
for (i, j) in A:
    for t in T:
        model.addConstr(x_flujo[i,j,t] <= K_flujo[i, j])


# ---------función objetivo--------------------------------
model.setObjective(quicksum(PM[r,t] for r in R for t in T), GRB.MINIMIZE)

# ---------resolver modelo---------------------------------
model.optimize()

# ---------imprimir resultados-----------------------------
# Abrir archivo de resultados
with open("resultados.txt", "w", encoding="utf-8") as f:
    f.write("---------- Manejo de Soluciones ----------\n\n")

    if model.Status == GRB.OPTIMAL:
        # (a) Valor óptimo del problema
        f.write(f"(a) Valor óptimo del problema: {model.ObjVal}\n\n")

        # (b) Agua aplicada por relave y por mes
        f.write("(b) Agua aplicada por relave y por mes:\n")
        for r in R:
            total_agua_r = sum(x_agua[r,t].X for t in T)
            f.write(f"Relave {r}, agua total aplicada: {total_agua_r}\n")
            for t in T:
                f.write(f"  Mes {t+1}: agua={x_agua[r,t].X}, cubierta={x_cubierta[r,t].X}, PM={PM[r,t].X}\n")
        f.write("\n")

        # (c) Flujo total desde cada fuente y por arco
        f.write("(c) Flujo total desde cada fuente y por arco:\n")
        for f_source in F:
            total_fuente = sum(x_flujo[i,j,t].X for (i,j) in A if i==f_source for t in T)
            f.write(f"Fuente {f_source}, flujo total enviado: {total_fuente}\n")
        for (i,j) in A:
            total_arco = sum(x_flujo[i,j,t].X for t in T)
            f.write(f"Arco {i}->{j}, flujo total: {total_arco}\n")
        f.write("\n")

        # (d) Cubierta vegetal activa y mantenciones
        f.write("(d) Cubierta vegetal y mantenciones por relave:\n")
        for r in R:
            total_cubierta = sum(z_veg[r,t].X for t in T)
            total_mant = sum(y_mant[r,t].X for t in T)
            f.write(f"Relave {r}: cubierta activa {total_cubierta} meses, mantenciones {total_mant} meses\n")
        f.write("\n")

        # (e) Validación de factibilidad y consistencia
        f.write("(e) Validación de factibilidad y consistencia:\n")
        
        # Presupuesto total
        presupuesto_total = sum(C_agua[r,t]*x_agua[r,t].X + C_vegmant[r,t]*z_veg[r,t].X + C_veginst[r,t]*y_veg[r,t].X 
                                for r in R for t in T) + sum(C_flujo[i,j]*x_flujo[i,j,t].X for (i,j) in A for t in T)
        if presupuesto_total > B_max:
            f.write(f"Presupuesto excedido: {presupuesto_total} > {B_max}\n")
        else:
            f.write(f"Presupuesto dentro del límite: {presupuesto_total} <= {B_max}\n")

        # PM máximo
        for r in R:
            for t in T:
                if PM[r,t].X > PM_max[r,t]:
                    f.write(f"PM máximo excedido en {r}, mes {t+1}: {PM[r,t].X} > {PM_max[r,t]}\n")

        f.write("\nLa solución es óptima")
    else:
        f.write("\nNo se encontró solución óptima.")

print("Resultados guardados en 'resultados.txt'")

if model.Status == GRB.INFEASIBLE:
    print("⚠️ El modelo es infactible. Generando archivo de diagnóstico...")
    model.computeIIS()
    model.write("infeasible_constraints.ilp")
    print("Archivo 'infeasible_constraints.ilp' generado. Revisa las restricciones conflictivas.")

