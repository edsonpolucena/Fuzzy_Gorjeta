# Implemente um sistema de inferência fuzzy para calcular o percentual de 
# gorjeta em um restaurante. O cálculo da gorjeta deve levar em consideração as seguintes regras:

# Se a refeição estiver insossa e o serviço ruim, a gorjeta será pouca

# Se a refeição estiver saborosa e o serviço excelente, a gorjeta será generosa

# Se o tempo de atendimento for demorado, não haverá gorjeta

# Se o tempo de atendimento for mediano ou rápido, haverá gorjeta

# Ver slides de Sistema de Inferência Fuzzy

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

refeicao = ctrl.Antecedent(np.arange(0, 11, 1), 'Refeicao')
servico = ctrl.Antecedent(np.arange(0, 11, 1), 'Servico')
tempo = ctrl.Antecedent(np.arange(0, 61, 1), 'Tempo')

gorjeta = ctrl.Consequent(np.arange(0, 11, 1), 'Gorjeta')

refeicao.automf(number=2, names=['insossa', 'saborosa'])
servico.automf(number=2, names=['ruim', 'excelente'])
tempo.automf(number=3, names=['demorado', 'mediano', 'rapido'])

gorjeta['pouca'] = fuzz.trimf(gorjeta.universe, [0, 0, 5])
gorjeta['media'] = fuzz.trimf(gorjeta.universe, [3, 5, 8])
gorjeta['generosa'] = fuzz.trimf(gorjeta.universe, [5, 10, 10])

regra1 = ctrl.Rule(refeicao['insossa'] & servico['ruim'], gorjeta['pouca'])
regra2 = ctrl.Rule(refeicao['saborosa'] & servico['excelente'], gorjeta['generosa'])
regra3 = ctrl.Rule(tempo['demorado'], gorjeta['pouca'])
regra4 = ctrl.Rule((tempo['mediano'] | tempo['rapido']), gorjeta['media'])

sistema_gorjeta = ctrl.ControlSystem([regra1, regra2, regra3, regra4])
simulador = ctrl.ControlSystemSimulation(sistema_gorjeta)

refeicao.view()
servico.view()
tempo.view()
gorjeta.view()

plt.show()

simulador.input['Refeicao'] = 8  
simulador.input['Servico'] = 9   
simulador.input['Tempo'] = 15   

simulador.compute()

print(f"Gorjeta recomendada: {simulador.output['Gorjeta']:.2f}")
