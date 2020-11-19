#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 18:24:30 2020

@author: felix
"""


class Prio_Table:
    '''
    Priority table class, used for incrementally merging an instructions
    dict. Entries are replaced or merged in the order specified
    by the following:
        (a) the order argument in keys: If the key takes the form 'akey @ N'
            with N an integer, N is used as the priority value, and the key
            is reduced to 'akey'.
        (b) the order parameter in `add`
        (c) the order of calls to `add`.
    The values in (a) and (b) default to 0 and are summed to yield the overall
    item priority.
    A finalised merge of all added entries can be retrieved with `get`.

    Parameters
    ----------
    appending : bool
        Defines how key collisions are managed. If True,
        entries are concatenated. If False, the highest-priority entry
        takes precedence.
    name : str, optional
        The name of the dict, used for warnings. The default is ''.

    '''
    def __init__(self, appending, name = ''):
        self.table = {}
        self.separators = {}
        self.types = {}
        self.append = appending
        self.name = name

    def add(self, key, value, warn = False, context = 'add', priority = 0, sep = '\n'):
        '''
        Adds a dict to the table

        Parameters
        ----------
        key : any valid dict key
            Entry key.
            String keys can specify priority by taking the form
            'key @ priority'.
        value :
            The value to be stored under this key.
        warn : bool, optional
            Whether to warn about replaced entries. The default is False.
        context : str, optional
            Used in replacement warnings. The default is 'add'.
        priority : int, optional
            Priority. This is added to the key-specified priority to yield the
            item's overall priority. The default is 0.
        sep : optional
            Separator to add between items when keys collide and the Prio_Table
            is in appending mode. Note that separators are key-specific, and
            the latest provided separator for any key is also used during `get`.
            The default is '\n'.
        '''
        key, prio = self.get_prio(key)
        prio += priority

        if key in self.types and type(value) != self.types[key]:
            value = self.types[key](value)
        self.separators[key] = sep

        if prio not in self.table:
            self.table[prio] = {key: value}
        elif key in self.table[prio]:
            if self.append:
                self.table[prio][key] = \
                    self.table[prio][key] + sep + value
            else:
                if warn and key in self.table[prio]:
                    print("Warning: Replacing {k}:{oldv} with {newv} from {context} in {name}".format(
                        k=key, oldv=self.table[prio][key], newv=value,
                        context=context, name=self.name))
                self.table[prio][key] = value
        else:
            self.table[prio][key] = value

    def ensure_type(self, key, a_type):
        '''
        Ensures all items under the given key are of the same type. Note that
        this is enforced both retroactively and proactively on future additions.

        Parameters
        ----------
        key :
            The key to enforce a type on
        a_type : type, None
            The type to enforce. If None, future additions will no longer be
            type-controlled.

        '''
        if a_type == None:
            self.types.pop(key, None)
            return

        self.types[key] = a_type
        for p, d in self.table.items():
            if key in d and type(d[key]) != a_type:
                d[key] = a_type(d[key])

    def get(self, warn = False, context = 'get'):
        '''
        Merge entries into a single dict, resolving by priority.

        Parameters
        ----------
        warn : bool, optional
            Whether to warn about replaced entries. The default is False.
        context : str, optional
            Used in replacement warnings. The default is 'get'.

        Returns
        -------
        dict
            A merged dictionary with unique keys, with entries resolved by
            the priority given during `add`.

        '''
        if len(self.table) == 0:
            return self.table
        table = self.table
        self.table = {}
        for p in sorted(table):
            for k,v in table[p].items():
                self.add(k, v, warn, context, sep = self.separators[k])
        merged = self.table[0]
        self.table = table
        return merged

    @staticmethod
    def get_prio(key):
        if type(key) == str and '@' in key:
            key, prio = key.split('@')
        else:
            prio = 0
        return key.strip(), int(prio)