from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

#Definicao da estrutura do modelo. Vinculos 
model = BayesianNetwork([('H', 'C'), ('I', 'C'), ('C', 'S')])

#Criação das tabelas de probabilidade condicional
#variable_card define o número de valores possíveis que a variável pode assumir
#evidence_card faz a mesma coisa mas para a evidência (a quem a variavel depende)

cpd_h = TabularCPD(variable='H', variable_card=2, values=[[0.15], [0.85]])

cpd_i = TabularCPD(variable='I', variable_card=2, values=[[0.46], [0.54]])

cpd_c = TabularCPD(variable='C', 
                   variable_card=2, 
                   values=[[0.99, 0.96, 0.95, 0.85 ], 
                           [0.01, 0.04, 0.05, 0.15 ]], 
                   evidence=['H', 'I'], 
                   evidence_card=[2, 2])


cpd_s = TabularCPD(variable='S', variable_card=2, values=[[0.09, 0.0], [0.91, 1]], evidence=['C'], evidence_card=[2])

#Associando CPD ao modelo
model.add_cpds(cpd_h, cpd_i, cpd_c, cpd_s)

print(model.get_cpds('S'))



# Realizando a inferência no modelo
inferencia = VariableElimination(model)

# Consultando a probabilidade conjunta P(H, I, C, S)
prob = inferencia.query(variables=['H'], evidence={'H': 1})

print(prob)

