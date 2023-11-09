import pandas

if __name__ == '__main__':
    document = pandas.read_xml('data/dev.xml')
    document.to_csv('data/dev.csv', index=False)