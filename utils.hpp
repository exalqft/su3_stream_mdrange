#pragma once

#include <concepts>

template <std::integral T>
T gcd(T a, T b)
{

    while (b != 0) {
        T temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

template <std::integral T>
T lcd(T a, T b)
{
    const T min = std::min(a, b);
    for (T i = 2; i <= min; ++i) {
        if (a % i == 0 && b % i == 0) {
            return i;
        }
    }
    return 1;
}
