import threading
import time

from ugot import ugot

import ns_controller
import ns_gui
import ns_shared


def main():
    SBBot = ugot.UGOT()
    ENGBot = ugot.UGOT()

    QueueChannels = ns_shared.QueueChannels()
    SharedState = ns_shared.SharedState()

    gui = ns_gui.GUI()
    process_manager = ns_controller.ProcessManager(
        SBBot, ENGBot, QueueChannels, SharedState
    )

    threads = [
        threading.Thread(target=process_manager.mainloop()),
    ]

    for _ in threads:
        _.start()

    gui.mainloop()  # ensure pygame runs in the main thread

    QueueChannels.kill_flag.set()  # signal all threads to die

    # now wait for everyone to die, patiently

    for _ in threads:
        _.join()

    # now everyone's dead, rip main! bye im out


if __name__ == "__main__":
    main()
