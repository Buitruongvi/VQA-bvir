def LoadData(path):
  data = []
  path = path
  with open(path, "r") as f:
    lines = f.readlines()
    for line in lines:
      temp = line.split('\t')
      qa = temp[1].split('?')

      if len(qa) == 3:
        answer = qa[2].strip()
      else:
        answer = qa[1].strip()

      data_sample = {
          'image_path': temp[0][:-2],
          'question': qa[0]+ '?',
          'answer': answer
      }

      data.append(data_sample)
  return data