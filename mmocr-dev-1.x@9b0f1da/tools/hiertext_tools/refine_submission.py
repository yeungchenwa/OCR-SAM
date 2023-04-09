import json


def convert_float_to_int(vertices):
    return [[int(x) for x in lst] for lst in vertices]


def find_vertices(data):
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "vertices":
                data[key] = convert_float_to_int(value)
            else:
                find_vertices(value)
    elif isinstance(data, list):
        for item in data:
            find_vertices(item)


if __name__ == '__main__':
    with open("line-predictions.jsonl", 'r') as file:
        data = json.loads(file.read())
    # change the value of vertices from float to int
    find_vertices(data)

    # add a empty prediction for 345ffd49d67797e1.jpg
    data["annotations"].append({
        "image_id":
        "345ffd49d67797e1",
        "paragraphs": [{
            'lines': [{
                'text':
                '',
                'words': [{
                    'text': '',
                    'vertices': [[0, 0], [0, 0], [0, 0], [0, 0]]
                }]
            }]
        }]
    })
    with open("line-predictions-refine.jsonl", "w") as file:
        json.dump(data, file)