
import restee as ree
from google_auth_oauthlib.flow import Flow
import json

# Build the `client_secrets.json` file by borrowing the
# Earth Engine python authenticator.
client_secrets = {
    'web': {
        'client_id': oauth.CLIENT_ID,
        'client_secret': oauth.CLIENT_SECRET,
        'redirect_uris': [oauth.REDIRECT_URI],
        'auth_uri': 'https://accounts.google.com/o/oauth2/auth',
        'token_uri': 'https://accounts.google.com/o/oauth2/token'
    }
}

# Write to a json file.
client_secrets_file = 'client_secrets.json'
with open(client_secrets_file, 'w') as f:
  json.dump(client_secrets, f, indent=2)

# Start the flow using the client_secrets.json file.
flow = Flow.from_client_secrets_file(client_secrets_file,
                                     scopes=oauth.SCOPES,
                                     redirect_uri=oauth.REDIRECT_URI)

# Get the authorization URL from the flow.
auth_url, _ = flow.authorization_url(prompt='consent')

# Print instructions to go to the authorization URL.
oauth._display_auth_instructions_with_print(auth_url)
print('\n')

# The user will get an authorization code.
# This code is used to get the access token.
code = input('Enter the authorization code: \n')
flow.fetch_token(code=code)

# Get an authorized session from the flow.
session = flow.authorized_session()



"""
Test solution

"""

session_ree = ree.EESession("<CLOUD-PROJECT>", "<PATH-TO-SECRET-KEY>")
states = ee.FeatureCollection('TIGER/2018/States')
maine = states.filter(ee.Filter.eq('NAME', 'Maine'))

# get a domain for the state of Maine at ~500m resolution
domain = ree.Domain.from_ee_geometry(session, maine, resolution=0.005)
