import json

class Coord:
  def __init__(self):
    self.x=0.0
    self.y=0.0
   
    

  def __init__(self,x,y):
    self.x = x
    self.y = y


class Landmark:
  def __init__(self):
    self.type = ""
    self.coords = []

  def __init__(self,type):
    self.type = type
    self.coords = []
  
  def AddType(self,type):
    self.type = type

  def AddCoords(self,x,y):
    coord = Coord(x,y)
    self.coords.append(coord)


class Stat:
  def __init__(self):
    self.timestamp = ""
    self.landmark = []

  def AddTimestamp(self,timestamp):
    self.timestamp = timestamp

  def AddLandmark(self,landmark):
    self.landmark.append(landmark)

  def ClearStat(self):
    self.timestamp=""
    for l in self.landmark:
      l.coords.clear()
      l.type = ""
    self.landmark.clear()
  


  def toJSON(self):
    return json.dumps(self, default=lambda o: o.__dict__,indent=4, sort_keys=True)

