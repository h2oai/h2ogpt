import json
import os
import sqlite3
import uuid

from enums import LangChainMode


def set_userid(db1s, requests_state1, get_userid_auth, guest_name=''):
    force = requests_state1 and 'username' in requests_state1
    db1 = db1s[LangChainMode.MY_DATA.value]
    assert db1 is not None and len(db1) == length_db1(), "%s %s" % (len(db1), length_db1())
    if db1[1] is None or force:
        db1[1] = get_userid_auth(requests_state1, id0=db1[1])
    if force or len(db1) == length_db1() and not db1[2]:
        username1 = None
        if 'username' in requests_state1:
            username1 = requests_state1['username']
            if username1 == guest_name:
                username1 += ':' + str(uuid.uuid4())
                requests_state1['username'] = username1
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


def create_table(auth_filename):
    conn = sqlite3.connect(auth_filename)
    cursor = conn.cursor()

    # Create table if not exists
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Users (
        username VARCHAR(255) PRIMARY KEY,
        data TEXT
    );
    """)
    conn.commit()
    conn.close()


def fetch_user(auth_filename, username, verbose=False):
    # Connect to an SQLite database (change the database path as necessary)
    if auth_filename.endswith('.json'):
        json_filename = auth_filename
        db_filename = auth_filename[:-4] + '.db'
    else:
        assert auth_filename.endswith('.db')
        db_filename = auth_filename
        json_filename = auth_filename[:-3] + '.json'

    if os.path.isfile(db_filename) and os.path.getsize(db_filename) == 0:
        os.remove(db_filename)
    if os.path.isfile(json_filename) and os.path.getsize(json_filename) == 0:
        os.remove(json_filename)

    if os.path.isfile(json_filename) and not os.path.isfile(db_filename):
        # then make, one-time migration
        with open(json_filename, 'rt') as f:
            auth_dict = json.load(f)
        create_table(db_filename)
        upsert_auth_dict(db_filename, auth_dict, verbose=verbose)
        # Slow way:
        # [upsert_user(db_filename, username1, auth_dict[username1]) for username1 in auth_dict]
    elif not os.path.isfile(db_filename):
        create_table(db_filename)

    if username in [None, '']:
        return {}

    conn = sqlite3.connect(db_filename)
    cursor = conn.cursor()

    try:
        # Prepare SQL query to fetch user data for a given username
        cursor.execute("SELECT data FROM Users WHERE username = ?", (username,))

        # Fetch the result
        result = cursor.fetchone()

        if result:
            # Deserialize the JSON string to a Python dictionary
            user_details = json.loads(result[0])
            assert isinstance(user_details, dict)
            return {username: user_details}
        else:
            return {}
    except Exception as e:
        print(f"An error occurred: {e}")
        return {}
    finally:
        # Close the database connection
        conn.close()


def upsert_user(db_filename, username, user_details, verbose=False):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_filename)
    cursor = conn.cursor()

    # Serialize the user_details dictionary to a JSON string
    data_string = json.dumps(user_details)

    # Prepare the UPSERT SQL command
    sql_command = """
    INSERT INTO Users (username, data) 
    VALUES (?, ?)
    ON CONFLICT(username) 
    DO UPDATE SET data = excluded.data;
    """

    try:
        # Execute the UPSERT command
        cursor.execute(sql_command, (username, data_string))
        conn.commit()  # Commit the changes to the database
        if verbose:
            print(f"User '{username}' updated or inserted successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Close the database connection
        conn.close()


def upsert_auth_dict(db_filename, auth_dict, verbose=False):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_filename)
    cursor = conn.cursor()

    # Serialize the user_details dictionary to a JSON string
    try:
        for username, user_details in auth_dict.items():
            data_string = json.dumps(user_details)

            # Prepare the UPSERT SQL command
            sql_command = """
            INSERT INTO Users (username, data) 
            VALUES (?, ?)
            ON CONFLICT(username) 
            DO UPDATE SET data = excluded.data;
            """

            # Execute the UPSERT command
            cursor.execute(sql_command, (username, data_string))
            if verbose:
                print(f"User '{username}' updated or inserted successfully.")
        conn.commit()  # Commit the changes to the database
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Close the database connection
        conn.close()


def get_all_usernames(auth_filename):
    assert auth_filename.endswith('.db'), "Bad auth_filename: %s" % auth_filename
    if not os.path.isfile(auth_filename):
        return []

    conn = sqlite3.connect(auth_filename)
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT username FROM Users")
        usernames = [row[0] for row in cursor.fetchall()]
        return usernames
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
    finally:
        conn.close()


def merge_dicts(original, updates):
    """
    Merge updates into the original dictionary. If a key points to a list, append the values.
    If a key points to a dictionary, merge the dictionaries.
    """
    for key, value in updates.items():
        if key in original:
            if isinstance(original[key], list) and isinstance(value, list):
                original[key].extend(value)
            elif isinstance(original[key], dict) and isinstance(value, dict):
                original[key] = merge_dicts(original[key], value)
            else:
                original[key] = value
        else:
            original[key] = value
    return original


def append_to_users_data(auth_filename, updates, verbose=False):
    assert auth_filename.endswith('.db'), "Bad auth_filename: %s" % auth_filename
    db_filename = auth_filename
    assert os.path.isfile(db_filename), "Database file %s does not exist." % db_filename

    conn = sqlite3.connect(db_filename)
    cursor = conn.cursor()

    try:
        # Fetch all usernames and their data
        cursor.execute("SELECT username, data FROM Users")
        users = cursor.fetchall()

        for username, data_string in users:
            user_details = json.loads(data_string)

            # Merge updates into user details
            user_details = merge_dicts(user_details, updates)

            # Serialize the updated user_details dictionary to a JSON string
            updated_data_string = json.dumps(user_details)

            # Prepare the UPSERT SQL command
            sql_command = """
            INSERT INTO Users (username, data)
            VALUES (?, ?)
            ON CONFLICT(username)
            DO UPDATE SET data = excluded.data;
            """

            # Execute the UPSERT command
            cursor.execute(sql_command, (username, updated_data_string))
            if verbose:
                print(f"User '{username}' updated successfully.")

        conn.commit()  # Commit the changes to the database
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        conn.close()


def append_to_user_data(auth_filename, username, updates, verbose=False):
    assert auth_filename.endswith('.db'), "Bad auth_filename: %s" % auth_filename
    db_filename = auth_filename
    assert os.path.isfile(db_filename), "Database file %s does not exist." % db_filename

    conn = sqlite3.connect(db_filename)
    cursor = conn.cursor()

    try:
        # Fetch the user data for the specified username
        cursor.execute("SELECT data FROM Users WHERE username = ?", (username,))
        user_data = cursor.fetchone()

        if not user_data:
            # Create new user details if user does not exist
            user_details = updates
            if verbose:
                print(f"User '{username}' does not exist in the database. Creating new user.")
        else:
            user_details = json.loads(user_data[0])
            # Merge updates into user details
            user_details = merge_dicts(user_details, updates)

        # Serialize the updated user_details dictionary to a JSON string
        updated_data_string = json.dumps(user_details)

        # Prepare the UPSERT SQL command
        sql_command = """
        INSERT INTO Users (username, data)
        VALUES (?, ?)
        ON CONFLICT(username)
        DO UPDATE SET data = excluded.data;
        """

        # Execute the UPSERT command
        cursor.execute(sql_command, (username, updated_data_string))
        if verbose:
            print(f"User '{username}' updated successfully.")

        conn.commit()  # Commit the changes to the database
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        conn.close()
