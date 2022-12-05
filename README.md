<h1>Intelligent Placer</h1>
<h2>Описание программы</h2>
  <p>Intelligent Placer должен по фотографиям предметов на светлой
поверхности и многоугольнику выдавать результат о том, можно ли расположить
данные предметы так, чтобы они влезли в нарисованную фигуру. Программа выдаёт True
, если объекты можно поместить в фигуру, иначе False</p>
<h2>Требования</h2>
<h3>Предметы</h3>
  <ul>
    <li>Объект не выходит за границы листа(на исходных данных)</li>
    <li>Лист виден целиком</li>
    <li>Предметы находятся на некотором расстоянии от границ листа и друг от друга(на тестовых данных)</li>
    <li>Объекты не могут перекрывать друг друга(на тестовых данных)</li>
    <li>Объекты должны находиться снизу листа(на тестовых данных)</li>
    <li>Предметы не могут находиться внутри друг друга</li>
  </ul>
<h3>Фигура</h3>
  <ul>
    <li>Выпуклый многоугольник</li>
    <li>Вершин должно быть столько, чтобы они были различимы на фотографии</li>
    <li>Рисуется на листе тёмным маркером</li>
  </ul>
<h3>Фотографии</h3>
  <ul>
    <li>Фон светлый</li>
    <li>Фотографии делаются в портретной ориентации</li>
    <li>Освещение комнатное(равномерное и без тёмных областей)</li>
    <li>Делаются на высоте 30-60 см от поверхности</li>
    <li>Угол между перпендикуляром к поверхности и направлением камеры < 5 градусов</li>
    <li>Формат ".jpg" / ".jpeg"</li>
    <li>Минимальное разрешение 10мп</li>
    <li>Предметы на фотографии не могут повторяться</li>
    <li>Лист формата А4</li>
    <li>Лист повёрнут горизонтально</li>
    <li>Размер изображения стандартный(в тестовых примерах они 806x1080px)</li>
  </ul>
<h2>План решения задачи</h2>
<p>Что сделано на данный момент. Наверняка будут какие-то серьёзные изменения</p>
<h3>Определение многоугольника</h3>
    <ol>
        <li>Выделить границы многоугольника фильтром Кэнни(для определения того, что это граница именно многоугольника есть две идеи)</li>
            <ul>
                <li>Т.к. лист помещается на фотографии целиком, то можно найти границы листа, и затем уже сам многоугольник, который будет внутри границ листа</li>
                <li>Учитывая требование, при котором объекты должны находиться ниже листа, можно попытаться 
искать самый верхний контур или же смотреть контуры в некоторой части фотографии(т.е., например, 3/4 области изображения, если считать сверху )</li>
            </ul>
        <li>Возможно дополнительная обработка(скорее всего, это будут морфологические операции)</li>
    </ol>
<h3>Выделение предметов</h3>
    <ol>
        <li>Выделяем границы предметов фильтром Кэнни(так же несколько вариантов)</li>
            <ul>
                <li>Основываемся на том же факте, что всё, что вне листа, будет предметами</li>
                <li>Используем требование, что предметы должны быть внизу листа, ищем контуры в некоторой области изображения(например, 1/4 часть листа, если считать снизу)</li>
                <li>Если мы знаем, какой и где контур многоугольника, то контурами предметов будут все полученные контуры, за исключением собственно многоугольника</li>
            </ul>
        <li>Так же, как и в случае многоугольника, возможна дополнительная обработка, дабы убрать шумы</li>
    </ol>
<h3>Проверка на вместимость всех предметов в многоугольник</h3>
<h4><b>САМЫЙ СТАРЫЙ ПЛАН</b></h4>
<p>Самый первый алгоритм был - это просто рандомно воозвращаемое True или False + проверка на сумму площадей</p>
<h4><b>СТАРЫЙ ПЛАН</b></h4>
    <blockquote>Пока что оставил этот пункт, потому что он немного поясняет идею, которая <s>сильно</s> несильно модифицированна + к тому же, что, я зря печатал его. Этот пункт можно спокойно пропустить</blockquote>
    <p>Именно этот пункт сделан максимально странным образом и наверняка будет изменён(уже)</p>
    <p>Суть идеи основана на комбинации эвристического алгоритма и обычных минимизирующих методов. Своими глазами вы можете увидеть псевдосимбиоз генетического алгоритма и встроенных методов "scipy.optimize".</p>
    <p>Собственно, на данный момент это работает примерно так:</p>
    <ol>
      <li>Вначале откидываем критичные случаи,например, сравнение по площадям, чтобы не ждать и не делать лишних действий</li>
      <li>Затем идёт перебор всех предметов с попыткой уместить их в многоугольник, если хоть один из них не влез, можно достойно сказать, что дальше мы ничего делать не будем и возращаем False. Если же удалось упаковать, то вырезаем этот контур из многоугольника и продолжаем итерации цикла до тех пор, пока не закончатся предмемты или же мы не сможем какой-то из них упаковать </li>
    </ol>
    <p>Чуть подробнее про момент, связанный именно с упаковкой предметов. По сути, фигурируют два значимых блока, которые работают(а иногда и не работают, но вы этого не видели) довольно диковато. Первый - это генетический алгоритм, который пытается подобрать начальную позицию в многоугольнике, чтобы оттуда уже начинать вмещать предмет. Второй - это минимизирующая функция, которая как бы уточняет местоположение предмета в многоугольнике. В итоге, после того, как мы получили несколько результатов, генетический алгоритм выбирает лучшую позицию, куда мы и размещаем предмет.</p>
    <p>Теперь ещё подробнее про каждый из блоков. Блок с минимизирующей функцией получает на вход многоугольник, предмет и начальные положения поворота, смещения по иксу и игреку. Мы "типа" пытаемся найти такие параметры поворота, смещений предмета, чтобы расстояние между предметов и многоугольником получилось наименьшее. Те этим мы стараемся укладывать предметы ближе к границе</p>
    <p>Блок с генетическим алгоритмом как раз использует блок с минимизирующей функцией, но при этом добавляется критерий качества полученной позиции. А именно - число точек контура, расстояние между которыми и многоугольником меньше заданной величины. Т.е чем больше точек, расстояние между которыми и многоугольником мало, тем плотнее к границе мы уложили предмет. Начальные позиции(которые как бы являюется индивидумом популяции), генерируются рандомно из диапазона точек многоугольника. Таким образом, мы находим точку, в которую может уместиться предмет. Если не может, то пробуем снова, пока не попытается N-ое число раз.Иначе просто считаем, что ничего не получилось. Конкретно этот момент будет дорабатываться и возможно вообще весь алгоритм поменяется</p>

  <h4><b>НОВЫЙ ПЛАН</b></h4>
  Здесь вместо генетического алгоритма используется метод дифференциальной эволюции. Суть в том, что мы получаем начальную позицию только для того, чтобы сместить предмет по ближе к многоугольнику, а затем уже запускаем метод scipy.optimize.differential_evolution, который использует идеи генетического алгоритма и позволяет получить минимум функции, которая определяет "оптимальность позиции". Т.е. если кратко, то алгоритм такой:
  <ol>
    <li>Отсекаем граничные случаи(нет предметов, площади предметов по сумме > площади многоугольника)</li>
    <li>Идём в цикле по каждому предмету</li>
    <li>Находим начальную позицию предмета</li>
    <li>С помощью scipy.optimize ищем оптимальную позицию. Функцию запускаем несколько раз(переменная N_ITER), тк есть случайность и дабы её минизировать из N_ITER результатов выбираем наилучший</li>
    <li>Если в итоге предмет не поместился, можно заканчивать и возвращать False</li>
    <li>Повторяем с 3 пункта действия до тех пор, пока не закончатся предметы или не сможем уместить какой-то</li>
    <li>Если смогли успешно уместить все предметы, возращаем True</li>
  </ol>
<h3>Оценка качества результатов</h3>
Если посмотреть на тестовые данные, то алгоритм справился довольно неплохо(95% верных результатов и один тест довольно странный). По картинкам упаковки видно, что алгоритм старается упаковывать так, как задумывалось, а именно с наибольшей плотностью соприкосновения с границей многоугольника и на максимальном удалении от центра.
<h3>Идеи по дальнейшему улучшению</h3>
На самом деле, идей достаточно много, вплоть до радикальной замены алгоритма. Но что касается именно реализации на данный момент, то первым делом можно оптимизировать вычисление расстояние между точками, ибо это слишком затратная по времени и мощностям операция. Дополнительно имеет смысл как-то модифицировать функцию, чтобы учитывались ещё какие-то факторы при упаковке и так же учесть комбинацию предметов при упаковке.