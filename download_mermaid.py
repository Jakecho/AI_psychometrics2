import base64
import requests
import time

def get_mermaid_image(mmd_file, out_file):
    with open(mmd_file, 'r') as f:
        graph = f.read()
    
    b64 = base64.b64encode(graph.encode('utf-8')).decode('utf-8')
    url = f"https://mermaid.ink/img/{b64}?bgColor=white"
    
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                with open(out_file, 'wb') as f:
                    f.write(resp.content)
                print(f"Successfully downloaded {out_file}")
                return
            else:
                print(f"Failed to fetch {out_file}, status code: {resp.status_code}")
        except Exception as e:
            print(f"Error fetching {out_file}: {e}")
        time.sleep(2)

if __name__ == "__main__":
    get_mermaid_image("venn_flow.mmd", "venn_flow.png")
