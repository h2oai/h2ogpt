import uuid

from enums import LangChainMode


def set_userid(db1s, requests_state1, get_userid_auth):
    force = requests_state1 and 'username' in requests_state1
    db1 = db1s[LangChainMode.MY_DATA.value]
    assert db1 is not None and len(db1) == length_db1()
    if not db1[1] or force:
        db1[1] = get_userid_auth(requests_state1, id0=db1[1])
    if not db1[2] or force:
        username1 = None
        if 'username' in requests_state1:
            username1 = requests_state1['username']
        db1[2] = username1


def set_userid_direct(db1s, userid, username):
    db1 = db1s[LangChainMode.MY_DATA.value]
    db1[1] = userid
    db1[2] = username


def get_userid_direct(db1s):
    return db1s[LangChainMode.MY_DATA.value][1] if db1s is not None else ''


def get_username_direct(db1s):
    return db1s[LangChainMode.MY_DATA.value][2] if db1s is not None else ''


def get_dbid(db1):
    return db1[1]


def set_dbid(db1):
    # can only call this after function called so for specific user, not in gr.State() that occurs during app init
    assert db1 is not None and len(db1) == length_db1()
    if db1[1] is None:
        #  uuid in db is used as user ID
        db1[1] = str(uuid.uuid4())


def length_db1():
    # For MyData:
    # 0: db
    # 1: userid and dbid
    # 2: username

    # For others:
    # 0: db
    # 1: dbid
    # 2: None
    return 3
