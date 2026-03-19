import json

import numpy as np
from dotenv import load_dotenv
from turso_python import AsyncTursoConnection, AsyncTursoCRUD

load_dotenv()

async def landmark_values_to_db(values, symbol):
    values = json.dumps(values.tolist())
    print(values)
    async with AsyncTursoConnection() as conn:
        crud = AsyncTursoCRUD(conn)
        await crud.create("landmarks", {"symbol": symbol, "landmark_coords": values})

async def delete_last_insert():
    async with AsyncTursoConnection() as conn:
        crud = AsyncTursoCRUD(conn)
        last_id = await crud.read("landmarks", columns="max(id)")
        last_id = last_id["rows"][0][0]
        print(last_id)
        await crud.delete(table="landmarks", where="id = ?", args=[last_id])
        
async def fetch_training_data():
    async with AsyncTursoConnection() as conn:
        crud = AsyncTursoCRUD(conn)
        data = await crud.read(table="landmarks", columns="symbol, landmark_coords")
        values = np.array([np.array(json.loads(row[1])) for row in data["rows"]])
        labels = [row[0] for row in data["rows"]]

        return (values, labels)
