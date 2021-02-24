from tkinter import Tk, Canvas
from random import randint

class Bunka:
    def __init__(self, x,y):
        self.vzhled = platno.create_rectangle(x*VEL,y*VEL+50, (x+1)*VEL, (y+1)*VEL + 50, fill="white",width=0)
        self.cislo = 0

        #self.text = platno.create_text(x*VEL+VEL/2, y*VEL+VEL/2 + 50, text = "0", anchor = "c", fill="#888")

class Had:
    def __init__(self,x,y,delka,smer,barva,jmeno = "Anonym",hrac = False,nahoru = "",doprava="",dolu="",doleva=""):
        self.x = x
        self.y = y
        self.delka = delka
        self.smer = smer
        self.barva = barva
        self.jmeno = jmeno
        self.hrac = hrac
        self.nahoru = nahoru
        self.dolu = dolu
        self.doprava = doprava
        self.doleva = doleva
    def logika(self):
        #Vaše práce...
        #self.smer = nový směr
        self.smer = randint(1,4)
        

def pohyb():
    for had in hadi:
        #Volání počítačem ovládaných hadů
        if not had.hrac:
            had.logika()
            
        if had.smer == 1: had.y -= 1
        if had.smer == 2: had.x += 1
        if had.smer == 3: had.y += 1
        if had.smer == 4: had.x -= 1
        if had.x<0: had.x = POCET-1
        if had.y<0: had.y = POCET-1
        if had.x==POCET: had.x = 0
        if had.y==POCET: had.y = 0

        if sit[had.x][had.y].cislo > 0:
            hadi.remove(had)
            continue
        if sit[had.x][had.y].cislo == -1:
            had.delka += 1
            drobek()
        sit[had.x][had.y].cislo = had.delka + 1
        
        
        platno.itemconfig(sit[had.x][had.y].vzhled, fill = had.barva)


    for i in range(POCET):
        for j in range(POCET):
            if sit[i][j].cislo>0:
                sit[i][j].cislo -= 1
                #platno.itemconfig(sit[i][j].text, text = sit[i][j].cislo)
                if sit[i][j].cislo == 0:
                    platno.itemconfig(sit[i][j].vzhled, fill = "white")
                    
    okno.after(1000//FPS, pohyb)
        
def drobek():
    x = randint(0,POCET-1)
    y = randint(0,POCET-1)
    while sit[x][y].cislo!=0:
        x = randint(0,POCET-1)
        y = randint(0,POCET-1)
    sit[x][y].cislo = -1
    platno.itemconfig(sit[x][y].vzhled, fill = "orange")
    
def stisk(e):
    k = e.keysym
    for had in hadi:
        if had.hrac:
            if k == had.nahoru and had.smer!=3:
                had.smer = 1 
            if k == had.doprava and had.smer!=4:
                had.smer = 2
            if k == had.dolu and had.smer!=1:
                had.smer = 3
            if k == had.doleva and had.smer!=2:
                had.smer = 4
POCET = 40
VEL = 500/POCET
FPS = 20
DROBKY = 10
okno = Tk()
platno = Canvas(okno, height = 550, width=500)
platno.pack()

sit = []

for i in range(POCET):
    sit.append([])
    for j in range(POCET):
        sit[i].append(Bunka(i,j))

hadi = [
        Had(x = 10, y = 10, delka = 1, smer = 1, barva = "red", jmeno = "Jarda", hrac = True,
            nahoru = "Up", doprava = "Right", dolu = "Down", doleva = "Left"),
        Had(x = 5, y = 5, delka = 20, smer = 3, barva = "blue", jmeno = "Pepa", hrac = True,
            nahoru = "w", doprava = "d", dolu = "s", doleva = "a"),
        Had(x = 20, y = 20, delka = 1, smer = 1, barva = "black", jmeno = "Komp", hrac = False)
    ]
for i in range(DROBKY):
    drobek()
okno.bind("<Key>",stisk)
pohyb()
okno.mainloop()