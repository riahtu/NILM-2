#save data as xlsx
from nilmtk import DataSet
from data_port_download import download_dataport, _dataport_dataframe_to_hdf, view_database_tables, view_buildings
view_database_tables(
        'zejianz@nevada.unr.edu',
        'MZzYJLDn9wZ6ra9',
        'university'   # university or commercial
    )


redd = DataSet('data/redd.h5')
redd.set_window(start='06-07-2013', end='06-15-2013')
elec = redd.buildings[1].elec
from nilmtk.utils import print_dict

print_dict(redd.metadata)
APPLIANCES = ['fridge']
for i in APPLIANCES:
    print(elec[i].available_columns())
    app = elec[i]
    df = next(app.load())
    df.to_excel('redd.xlsx',sheet_name=i)


