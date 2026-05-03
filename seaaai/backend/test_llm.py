import requests
endpoints=['http://localhost:8000/health','http://localhost:8000/check-llm']
for e in endpoints:
    try:
        r = requests.get(e, timeout=5)
        print(e, r.status_code)
        print(r.text)
    except Exception as ex:
        print('ERR', e, ex)

payload={'prediction':'FINAL-TEST','confidence':0.7,'patientName':'Full Run','hospitalName':'Local Hosp','doctorName':'Dr Run'}
try:
    rr = requests.post('http://localhost:8000/llm-insight', json=payload, timeout=60)
    print('LLM_INSIGHT', rr.status_code)
    print(rr.text)
except Exception as ex:
    print('LLM POST ERR', ex)
