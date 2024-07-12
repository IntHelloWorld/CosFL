public class Outer {
    public static void main(String[] args) {
        Runnable r1 = new Runnable() {
            @Override
            public void run() {
                System.out.println("Anonymous class 1");
                Tool t1 = new Tool() {
                    @Override
                    public void do() {
                        System.out.println("Anonymous class 1.1");
                    }
                };
                Tool t2 = new Tool() {
                    @Override
                    public void do() {
                        System.out.println("Anonymous class 1.2");
                    }
                };
            }
        };

        Runnable r2 = new Runnable() {
            @Override
            public void run() {
                System.out.println("Anonymous class 2");
            }
        };

        Runnable r2 = new Runnable() {
            @Override
            public void run() {
                System.out.println("Anonymous class 2");
            }
        };
    }
}