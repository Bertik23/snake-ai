from tkinter import Tk, Canvas
from random import randint
import numpy as np
import traceback


class Bunka:
    def __init__(self, x, y, platno, VEL):
        self.vzhled = platno.create_rectangle(
            x*VEL, y*VEL+50, (x+1)*VEL, (y+1)*VEL + 50, fill="white", width=0)
        self.cislo = 0


class Had:
    def __init__(self, id, x, y, delka, smer, barva, jmeno="Anonym",
                 hrac=False, nahoru="", doprava="", dolu="", doleva=""):
        self.id = id
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
        pass
        # Vaše práce...
        # self.smer = nový směr
        # self.smer = randint(1,4)

    def setSmer(self, newSmer):
        try:
            if [3, 4, 1, 2][newSmer-1] != self.smer:
                self.smer = newSmer
        except IndexError as e:
            print(newSmer)
            print(traceback.format_exc())
            raise e

    def __repr__(self):
        return f"Had {self.jmeno}"


class Hra:
    def __init__(self):
        self.okno = Tk()
        self.platno = Canvas(self.okno, height=550, width=500)
        self.platno.pack()
        self.reset()

    def pohyb(self):
        reward = 0
        done = False
        for had in self.hadi:
            # Volání počítačem ovládaných hadů
            if not had.hrac:
                had.logika()
            if had.smer == 1:
                had.y -= 1
            if had.smer == 2:
                had.x += 1
            if had.smer == 3:
                had.y += 1
            if had.smer == 4:
                had.x -= 1
            if had.x < 0:
                had.x = self.POCET-1
            if had.y < 0:
                had.y = self.POCET-1
            if had.x == self.POCET:
                had.x = 0
            if had.y == self.POCET:
                had.y = 0

            if self.sit[had.x][had.y].cislo > 0:
                self.hadi.remove(had)
                if had.id == 1:
                    reward -= 200
                    done = True
                continue
            if self.sit[had.x][had.y].cislo == -1:
                if had.id == 1:
                    reward += 50
                had.delka += 1
                self.numOfDrobeks -= 1
                if self.numOfDrobeks < 10:
                    self.drobek()
            self.sit[had.x][had.y].cislo = had.delka + 1
            if had.id == 1:
                reward += 1

            self.platno.itemconfig(
                self.sit[had.x][had.y].vzhled, fill=had.barva)

        for i in range(self.POCET):
            for j in range(self.POCET):
                if self.sit[i][j].cislo > 0:
                    self.sit[i][j].cislo -= 1
                    # platno.itemconfig(sit[i][j].text, text = sit[i][j].cislo)
                    if self.sit[i][j].cislo == 0:
                        self.platno.itemconfig(
                            self.sit[i][j].vzhled, fill="white")
        return reward, done

    def drobek(self):
        x = randint(0, self.POCET-1)
        y = randint(0, self.POCET-1)
        while self.sit[x][y].cislo != 0:
            x = randint(0, self.POCET-1)
            y = randint(0, self.POCET-1)
        self.sit[x][y].cislo = -1
        self.platno.itemconfig(self.sit[x][y].vzhled, fill="orange")
        self.numOfDrobeks += 1

    def getAiHad(self):
        for had in self.hadi:
            if had.id == 1:
                return had
        else:
            return None

    def state(self):
        return np.append(
            [[had.x, had.y] for had in [self.getAiHad()]]
            if self.getAiHad() is not None else [-2, -2],
            np.array([[cell.cislo for cell in row] for row in self.sit])
            .flatten())

    def reset(self):

        self.sit = []
        self.n = 0

        self.POCET = 40
        self.VEL = 500/self.POCET
        self.FPS = 20
        self.DROBKY = 100
        self.numOfDrobeks = 0

        for i in range(self.POCET):
            self.sit.append([])
            for j in range(self.POCET):
                self.sit[i].append(Bunka(i, j, self.platno, self.VEL))

        def addHad(**hadKwargs):
            x = randint(0, self.POCET-1)
            y = randint(0, self.POCET-1)
            self.hadi = []
            while self.sit[x][y].cislo != 0:
                x = randint(0, self.POCET-1)
                y = randint(0, self.POCET-1)
            self.hadi.append(Had(x=x, y=y, **hadKwargs))
            # Had(id = 0, x = 20, y = 20, delka = 5, smer = 1,
            # barva = "black", jmeno = "Komp", hrac = False),
            # Had(id = 1, x = 30, y = 30, delka = 5, smer = 1,
            # barva = "green", jmeno = "Agent 007", hrac = True)

        addHad(id=0, delka=5, smer=1, barva="black", jmeno="Komp", hrac=False)
        addHad(id=1, delka=5, smer=1, barva="green",
               jmeno="Agent 007", hrac=True)

        for i in range(self.DROBKY):
            self.drobek()

        self.pohyb()

        return self.state()

    def step(self, action):
        # print(action, self.n, self.hadi)
        self.n += 1
        for i, had in enumerate(self.hadi):
            if had.id == 1:
                self.hadi[i].setSmer(action)
        reward, done = self.pohyb()
        return self.state(), reward, done, {}

    def render(self):
        self.okno.update()

    def close(self):
        pass
