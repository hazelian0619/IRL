import sys
sys.path.insert(0, 'external_town/reverie/backend_server')
try:
    from persona.persona import Persona
    print("OK: Import successful")
except Exception as e:
    print("ERROR: " + str(e))
