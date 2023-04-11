from gradio_client import Client

client = Client("http://localhost:7860")
print(client.view_api(all_endpoints=True))
res = client.predict("Who are you?", api_name='/instruction')
print(res)
