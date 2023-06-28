
import sqlite3


def addUser(id, name):
    con = sqlite3.connect('Amiko.db', check_same_thread=False)
    cur=con.cursor()
    if (cur.execute("SELECT IdUser FROM Users WHERE IdUser=?",(id,)).fetchone() is None) :
        con.close()
        return "Новый Пользователь"
    else:
        cur.execute("UPDATE Users SET Username=? WHERE IdUser=?", (name,id,))
        con.commit()
        con.close()
        return "Старый Пользователь"


