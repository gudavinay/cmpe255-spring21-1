import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Solution:
    def __init__(self) -> None:
        # TODO:
        # Load data from data/chipotle.tsv file using Pandas library and
        # assign the dataset to the 'chipo' variable.
        file = 'data/chipotle.tsv'
        self.chipo = pd.read_csv(file, sep='\t')

    def top_x(self, count) -> None:
        # TODO
        # Top x number of entries from the dataset and display as markdown format.
        topx = self.chipo.head(count)
        print(topx.to_markdown())

    def count(self) -> int:
        # TODO
        # The number of observations/entries in the dataset.
        return self.chipo.shape[0]

    def info(self) -> None:
        # TODO
        # print data info.
        return self.chipo.info()

    def num_column(self) -> int:
        # TODO return the number of columns in the dataset
        # print(self.chipo.shape[1])
        # for index in self.chipo.columns['index']
        #     print(index)
        return self.chipo.shape[1]

    def print_columns(self) -> None:
        # TODO Print the name of all the columns.
        # return self.chipo.columns.to_series()
        return self.chipo.columns.to_list()

    def most_ordered_item(self):
        # TODO
        a = self.chipo[['item_name','quantity','order_id']]
        # print(a)
        a = a.groupby('item_name',as_index=False)
        # print(a)
        a = a.sum()
        # print(a)
        a = a.sort_values(by=['quantity'],ascending = False)
        # print(a)
        a = a.head(1)
        # print(a)
        item_name = a['item_name'].values[0]
        order_id = a['order_id'].values[0]
        quantity = a['quantity'].values[0]
        return item_name, order_id, quantity

    def total_item_orders(self) -> int:
       # TODO How many items were orderd in total?
        return self.chipo['quantity'].sum()

    def total_sales(self) -> float:
        # TODO
        # 1. Create a lambda function to change all item prices to float.
        # 2. Calculate total sales.
        # df_list = self.chipo['item_price'].value_counts()
        # df_list2 = self.chipo['quantity'].to_list()
        # x = self.chipo
        # self.chipo['item_price'] = self.chipo['item_price'][1:-1]
        # print('***  ' + self.chipo['item_price'])
        # x= lambda a:a'

        a = self.chipo['item_price'].apply(lambda x:float(x.replace('$','')))
        aa = self.chipo['quantity'] * a
        # print(a)
        # pd.np.multiply(self.chipo['item_price'],)
        # sum = 0.0
        # for i in df_list:
        #     sum+=float(i.replace('$',''))
        # print(sum)
        return aa.sum()

    def num_orders(self) -> int:
        # TODO
        # How many orders were made in the dataset?
        # print(self.chipo['order_id'].tail(1))
        # print(self.chipo['order_id'].tail(1).iat[0])
        return self.chipo['order_id'].tail(1).iat[0]

    def average_sales_amount_per_order(self) -> float:
        # TODO
        # print(round(Solution.total_sales(self)/Solution.num_orders(self),2))
        return round(Solution.total_sales(self)/Solution.num_orders(self),2)

    def num_different_items_sold(self) -> int:
        # TODO
        # How many different items are sold?
        # self.chipo['item_name'].unique()
        # print(self.chipo['item_name'].unique().shape)
        return self.chipo['item_name'].unique().shape[0]

    def plot_histogram_top_x_popular_items(self, x: int) -> None:
        from collections import Counter
        letter_counter = Counter(self.chipo.item_name)
        # print(type(letter_counter))
        # print(letter_counter)
        # print(letter_counter.items())
        # print(pd.DataFrame(letter_counter.items()))
        # TODO
        # 1. convert the dictionary to a DataFrame
        # print(pd.DataFrame.from_dict(letter_counter))
        df = pd.DataFrame(letter_counter.items())
        # print(df)
        # 2. sort the values from the top to the least value and slice the first 5 items
        # print(df.sort_values(1, ascending=False).head(x))
        final = df.sort_values(1, ascending=False).head(x)
        # 3. create a 'bar' plot from the DataFrame
        final.plot.bar(x=0,y=1,rot=0)
        # 4. set the title and labels:
        #     x: Items
        #     y: Number of Orders
        #     title: Most popular items
        plt.xlabel('Items')
        plt.ylabel('Number of Orders')
        plt.suptitle('Most popular items')
        # 5. show the plot. Hint: plt.show(block=True).
        plt.show(block = True)
        # pass

    def scatter_plot_num_items_per_order_price(self) -> None:
        # TODO
        # 1. create a list of prices by removing dollar sign and trailing space.
        self.chipo['item_price'] = self.chipo['item_price'].apply(lambda x:float(x.replace('$','')))
        # print(self.chipo)
        # 2. groupby the orders and sum it.
        orderedChipo = self.chipo.groupby('order_id').sum()
        # print(orderedChipo)
        # a = self.chipo['item_price'].replace('$','')
        # print(a)

        # 3. create a scatter plot:
        #       x: orders' item price
        #       y: orders' quantity
        #       s: 50
        #       c: blue
        orderedChipo.plot.scatter(x='item_price',y='quantity',s=50,c='blue')
        plt.xlabel('Order Price')
        plt.ylabel('Num Items')
        plt.suptitle('Numer of items per order price')
        # 4. set the title and labels.
        #       title: Numer of items per order price
        #       x: Order Price
        #       y: Num Items
        plt.show(block = True)
        


def test() -> None:
    solution = Solution()
    solution.top_x(10)
    count = solution.count()
    print(count)
    assert count == 4622
    solution.info()
    print(solution.print_columns())
    count = solution.num_column()
    assert count == 5
    item_name, order_id, quantity = solution.most_ordered_item()
    assert item_name == 'Chicken Bowl'
    assert order_id == 713926
    assert quantity == 761
    total = solution.total_item_orders()
    assert total == 4972
    assert 39237.02 == solution.total_sales()
    assert 1834 == solution.num_orders()
    assert 21.39 == solution.average_sales_amount_per_order()
    assert 50 == solution.num_different_items_sold()
    solution.plot_histogram_top_x_popular_items(5)
    solution.scatter_plot_num_items_per_order_price()


if __name__ == "__main__":
    # execute only if run as a script
    test()
