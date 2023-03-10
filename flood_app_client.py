import sys
import asyncio
import websockets
import json
from source_data.GCPdata import GCP_IO
from flood_app import FLOOD_APP

from source_data.file_handler import FILE_HANDLER

class FLOOD_CLIENT(FLOOD_APP):

    #uri = 'ws://localhost:8765'
    uri = 'ws://34.173.99.39:8086'

    def __init__(self,*args):
        super().__init__(FILE_HANDLER(),*args)

    def update_sim_settings(self, **kwargs):
        super().update_sim_settings(**kwargs)
        
        if 'fileHandler' in kwargs:
            del kwargs['fileHandler']

        clientInputJson = json.dumps(kwargs)
        asyncio.run(self._send_simulation_settings(clientInputJson))

    def _button_pushed(self):
        asyncio.run(self._send_server_shutdown_request())

    def run_flood_simulation(self):
        htmlCode = asyncio.run(self._send_simulation_request())
        self._update_map_view(htmlCode=htmlCode)

    async def _send_simulation_settings(self,jsonData):
        async with websockets.connect(self.uri) as websocket:
            await websocket.send(jsonData)
        print('Input sent to server')

    async def _send_simulation_request(self):
        async with websockets.connect(self.uri) as websocket:
            await websocket.send('"run_sim"')
            print('request sent')
            mapHtmlJson = await websocket.recv()
        return json.loads(mapHtmlJson)

    async def _send_server_shutdown_request(self):
        async with websockets.connect(self.uri) as websocket:
            await websocket.send('"shutdown"')
            print('shutdown requested')


if __name__ == '__main__':
    appClient = FLOOD_CLIENT(sys.argv)
    appClient.run()