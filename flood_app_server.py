import sys
import asyncio
import websockets
from flood_app import FLOOD_APP
from source_data.GCPdata import GCP_IO
import json
from source_data.file_handler import FILE_HANDLER

'''
(https://stackoverflow.com/questions/62631176/python-socket-to-connect-over-global-public-ip-address)
in server you always use local IP, not external IP. Server has to bind to local network card (NIC - Network Internet Card) or to all of them (when you use 0.0.0.0). And client which want to connect from internet has to use external IP. Client connects to external IP which means IP of Internet Provider router, and router sends/redirects it to your server. – 
furas ('0.0.0.0') (probably local IP for VM is 10.128.3, see instance details)
 '''


class FLOOD_SERVER(FLOOD_APP):
    
    #IP = 'localhost'
    #port = 8765
    IP = '10.128.0.3'
    port = 8086

    def __init__(self,*args):
        super().__init__(FILE_HANDLER(),*args)
        self._loop = None

    def _init_QApplication(self,*args):
        pass

    def run(self):
        asyncio.run(self.web_serve())

    async def web_serve(self):
        async with websockets.serve(self.client_input,self.IP,self.port) as self._server:
            await asyncio.Future()

    async def client_input(self,websocket):
        clientInputJson = await websocket.recv()
        print('User input received')

        clientInput = json.loads(clientInputJson)
        print(clientInput)
        if type(clientInput) is dict:
            clientSimSettings = clientInput
            self.update_sim_settings(**clientSimSettings)
        elif 'run_sim' == clientInput:
            self.run_flood_simulation()
            await self._send_map_html(websocket)
        elif 'shutdown' == clientInput:
            self._shutdown()

    def _shutdown(self):
        print('Shutting down')
        #self._server.close()
        #await self._server.wait_closed()
        exit()

    def _init_main_window(self):
        pass
    def _init_text_fields(self):
        pass
    def _update_map_view(self, htmlCode=None):
        pass

    async def _send_map_html(self,websocket):
        await websocket.send(
            json.dumps(self.mapData.getvalue().decode())
            )


if __name__ == '__main__':
    appServer = FLOOD_SERVER(sys.argv)
    appServer.run()