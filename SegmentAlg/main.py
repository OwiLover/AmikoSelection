import threading
import time
import addDB2 as db
import Amiko as amiko
import yoloPyTorch as maincam

t1 = threading.Thread(target=maincam.maincam, args=[])


t1.daemon = True


t1.start()

t2 = threading.Thread(target=db.addBD, args=[])
t2.daemon = True
t2.start()

t1.join()
t2.join()

# time.sleep(30)
# t1.terminate()