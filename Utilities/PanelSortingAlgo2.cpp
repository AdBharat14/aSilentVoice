#include <bits/stdc++.h>

using namespace std;

vector<vector<int>> ans;

bool customComparatorHorz(vector<int> v1, vector<int> v2)
{
    return v1[0] > v2[0];
}

bool customComparatorVert(vector<int> v1, vector<int> v2)
{
    return v1[1] < v2[1];
}

void sortVert(vector<vector<int>> coord);

void sortHorz(vector<vector<int>> coord)
{
    // cout<<"hello";
    if (coord.size() == 0)
    {
        return;
    }
    if (coord.size() == 1)
    {
        // cout<<"hello";
        ans.push_back(coord[0]);
        return;
    }
    sort(coord.begin(), coord.end(), customComparatorHorz);

    // cout<<endl<<coord.size()<<endl;
    // for (auto &el : coord)
    // {
    //     for (auto &el2 : el)
    //     {
    //         cout << el2 << " ";
    //     }
    //     cout << endl;
    // }
    int i = 0;
    vector<int> visited(coord.size(), 0);
    int n = coord.size();
    while (i < n)
    {
        if (!visited[i])
        {
            vector<vector<int>> v;
            int flag = 1;
            int x = coord[i][0], x1 = coord[i][2];
            v.push_back(coord[i]);
            visited[i] = 1;
            while (flag)
            {
                flag = 0;
                for (int j = 0; j < n; j++)
                {
                    if (!visited[j])
                    {
                        if (x <= coord[j][2])
                        {
                            x = min(x, coord[j][0]);
                            v.push_back(coord[j]);
                            visited[j] = 1;
                            flag = 1;
                        }
                    }
                }
            }
            // cout<<endl<<v.size()<<endl;
            // for (auto &el : v)
            // {
            //     for (auto &el2 : el)
            //     {
            //         cout << el2 << " ";
            //     }
            //     cout << endl;
            // }

            if (v.size() == coord.size())
            {
                for (auto &el : v)
                {
                    ans.push_back(el);
                }
                return;
            }
            sortVert(v);
        }
        i += 1;
    }
}

void sortVert(vector<vector<int>> coord)
{
    if (coord.size() == 0)
    {
        return;
    }
    if (coord.size() == 1)
    {
        ans.push_back(coord[0]);
        return;
    }
    sort(coord.begin(), coord.end(), customComparatorVert);
    int i = 0;
    int n = coord.size();
    vector<int> visited(n, 0);
    while (i < n)
    {
        if (!visited[i])
        {
            vector<vector<int>> v;
            int flag = 1;
            int y = coord[i][1], y1 = coord[i][3];
            // cout<<y<<" "<<y1<<endl<<endl;
            v.push_back({coord[i][0], coord[i][1], coord[i][2], coord[i][3]});
            visited[i] = 1;
            while (flag)
            {
                flag = 0;
                for (int j = 0; j < n; j++)
                {
                    if (!visited[j])
                    {
                        if (y1 >= coord[j][1])
                        {
                            y1 = max(y1, coord[j][3]);
                            v.push_back(coord[j]);
                            visited[j] = 1;
                            flag = 1;
                        }
                    }
                }
            }

            // cout<<endl<<v.size()<<endl;
            // for (auto &el : v)
            // {
            //     for (auto &el2 : el)
            //     {
            //         cout << el2 << " ";
            //     }
            //     cout << endl;
            // }
            if (v.size() == coord.size())
            {
                for (auto &el : v)
                {
                    ans.push_back(el);
                }
                return;
            }
            sortHorz(v);
        }
        i += 1;
    }
}

int main(int argc, char *argv[])
{
    freopen("out.txt", "w", stdout);
    vector<vector<int>> coord;
    for (int i = 1; i < argc; i += 4)
    {
        int num1 = atoi(argv[i]), num2 = atoi(argv[i + 1]), num3 = atoi(argv[i + 2]), num4 = atoi(argv[i + 3]);
        coord.push_back({num1, num2, num3, num4});
    }
    // cout<<coord.size();
    // cout << "hi";
    sortVert(coord);
    // cout<<ans.size()<<endl;
    for (auto &el : ans)
    {
        for (auto &el2 : el)
        {
            cout << el2 << " ";
        }
        cout << endl;
    }
    // cout << "hi";
    return 0;
}
